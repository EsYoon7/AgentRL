from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from rich import get_console
from rich.console import Console
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (DataTable,
                             Footer,
                             Header,
                             OptionList,
                             ProgressBar,
                             RichLog,
                             Static,
                             TabbedContent,
                             TabPane)
from textual.widgets.data_table import CellDoesNotExist, DuplicateKey
from textual.widgets.option_list import Option
from textual._context import NoActiveAppError

from ..event.bus import EventBus
from ..event.types import (InitializedEvent,
                           MetricsEvent,
                           ResultListEvent,
                           SessionCompletedEvent,
                           SessionStartedEvent,
                           SpecsEvent,
                           TokenUsageSummaryEvent,
                           WorkflowCompletedEvent,
                           WorkflowStartEvent)
from ..session.types import MetricResult, MetricType
from ..utils import setup_rich_logging


class RichLogConsole(Console):

    def __init__(self, rich_log_widget: RichLog, **kwargs):
        super().__init__(**kwargs)
        self.rich_log_widget = rich_log_widget

    def print(self, *args):
        try:
            self.rich_log_widget.write(*args)
        except NoActiveAppError:
            get_console().print(*args)


class EvalApp(App):

    TITLE = 'agentrl-eval'
    CSS = '''
    #main {
        height: 1fr;
    }

    #tabs {
        height: 1fr;
    }

    #sessions_table {
        width: 1fr;
    }

    #metrics_table {
        width: 1fr;
    }

    #resume_prompt {
        padding: 1 2;
    }

    #status_bar {
        height: 2;
        color: $footer-foreground;
        background: $footer-background;
    }

    #progress Bar {
        width: 1fr;
    }

    #result_summary {
        width: auto;
    }

    #token_summary {
        width: 1fr;
        text-align: right;
    }
    '''

    # elements
    _tabs: TabbedContent
    _log_panel: RichLog
    _sessions_table: DataTable[Union[Text, str, int]]
    _metrics_table: DataTable[str]
    _resume_list: OptionList
    _progress_bar: ProgressBar
    _result_summary: Static
    _token_summary: Static
    _footer: Footer

    def __init__(self,
                 *,
                 event_bus: EventBus,
                 **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        self._result_summary_for_metric = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id='main'):
            self._tabs = TabbedContent(id='tabs')
            with self._tabs:
                with TabPane('Logs', id='tab_logs'):
                    self._log_panel = RichLog(id='log_panel', max_lines=10000, auto_scroll=False)
                    yield self._log_panel
                with TabPane('Sessions', id='tab_sessions'):
                    self._sessions_table = DataTable(id='sessions_table', cursor_type='row')
                    yield self._sessions_table
                with TabPane('Metrics', id='tab_metrics'):
                    self._metrics_table = DataTable(id='metrics_table', cursor_type='row')
                    yield self._metrics_table
                with TabPane('Resume', id='tab_resume'):
                    yield Static('Should we resume a previous evaluation?', id='resume_prompt')
                    self._resume_list = OptionList(id='resume_list')
                    yield self._resume_list
                with TabPane('Results', id='tab_results'):
                    yield Static('placeholder')  # TODO
            with Vertical(id='status_bar'):
                self._progress_bar = ProgressBar(total=100, id='progress')
                yield self._progress_bar
                with Horizontal():
                    self._result_summary = Static('Initializing...', id='result_summary')
                    yield self._result_summary
                    self._token_summary = Static('', id='token_summary')
                    yield self._token_summary
        self._footer = Footer(id='footer')
        yield self._footer

    async def on_mount(self):
        # initialize logging console
        console = RichLogConsole(self._log_panel)
        setup_rich_logging(console)

        # hide tabs by default
        self._tabs.hide_tab('tab_sessions')
        self._tabs.hide_tab('tab_metrics')
        self._tabs.hide_tab('tab_resume')
        self._tabs.hide_tab('tab_results')

        # add event listeners
        self.event_bus.subscribe(MetricsEvent, self._handle_metrics)
        self.event_bus.subscribe(TokenUsageSummaryEvent, self._handle_token_usage_summary)
        self.event_bus.subscribe(ResultListEvent, self._handle_result_list)
        self.event_bus.subscribe(SpecsEvent, self._handle_specs)
        self.event_bus.subscribe(SessionStartedEvent, self._handle_session_started)
        self.event_bus.subscribe(SessionCompletedEvent, self._handle_session_completed)
        self.event_bus.subscribe(WorkflowCompletedEvent, self._handle_workflow_completed)

        # emit initialized event
        await self.event_bus.publish(InitializedEvent())

    async def on_unmount(self):
        # cleanup event listeners
        self.event_bus.unsubscribe(MetricsEvent, self._handle_metrics)
        self.event_bus.unsubscribe(TokenUsageSummaryEvent, self._handle_token_usage_summary)
        self.event_bus.unsubscribe(ResultListEvent, self._handle_result_list)
        self.event_bus.unsubscribe(SpecsEvent, self._handle_specs)
        self.event_bus.unsubscribe(SessionStartedEvent, self._handle_session_started)
        self.event_bus.unsubscribe(SessionCompletedEvent, self._handle_session_completed)
        self.event_bus.unsubscribe(WorkflowCompletedEvent, self._handle_workflow_completed)

    async def on_tabbed_content_tab_activated(self, e: TabbedContent.TabActivated) -> None:
        if e.tabbed_content.id == 'tabs':
            if e.tab.id == 'tab_resume' or e.tab.id == 'tab_results':
                self._footer.display = True
            else:
                self._footer.display = False

    async def on_option_list_option_selected(self, e: OptionList.OptionSelected) -> None:
        if e.option_list.id == 'resume_list':
            self._result_summary.update('Initializing...')
            if e.option_id is None:
                self._tabs.active = 'tab_logs'
                self._tabs.hide_tab('tab_resume')
                await self.event_bus.publish(WorkflowStartEvent(path=None))
            else:
                self._tabs.active = 'tab_logs'
                self._tabs.hide_tab('tab_resume')
                await self.event_bus.publish(WorkflowStartEvent(path=Path(e.option_id)))

    async def _handle_metrics(self, e: MetricsEvent):
        name = 'SR' if e.metric_type == MetricType.SUCCESS_RATE else 'Avg'

        if len(e.items) > 1:
            # multiple models or tasks are present, make a table of results
            self._result_summary_for_metric = False
            table = self._metrics_table
            table.clear()

            models: set[str] = set()
            tasks: set[str] = set()
            for item in e.items:
                if item.model not in models:
                    models.add(item.model)
                if item.task not in tasks:
                    tasks.add(item.task)

            if len(models) > 1:
                for label, column_key in [
                    ('Model', 'model'),
                    *[(task, task) for task in tasks]
                ]:
                    try:
                        table.add_column(label, key=column_key)
                    except DuplicateKey:
                        pass
            else:
                for label, column_key in [
                    ('Task', 'task'),
                    *[(model, model) for model in models]
                ]:
                    try:
                        table.add_column(label, key=column_key)
                    except DuplicateKey:
                        pass

            rows: dict[str, dict[str, str]] = {}
            for item in e.items:
                if len(models) > 1:
                    if item.model not in rows:
                        rows[item.model] = {'model': item.model}
                elif item.task not in rows:
                    rows[item.task] = {'task': item.task}
                formatted = f'Valid: {item.valid}, {name}: {item.avg * 100:.2f}'
                if item.std is not None:
                    formatted += f' ± {item.std * 100:.2f}'
                if item.bon is not None:
                    formatted += f', BoN: {item.bon * 100:.2f}'
                if len(models) > 1:
                    rows[item.model][item.task] = formatted
                else:
                    rows[item.task][item.model] = formatted

            for row in rows.values():
                table.add_row(
                    *[row.get(col.value, '-') for col in table.columns],
                    height=1
                )

            if len(models) > 1:
                table.sort('model')
            else:
                table.sort('task')

            self._tabs.show_tab('tab_metrics')
            return

        if len(e.items) == 1:
            # only one metric, display a simple summary on status bar
            item = e.items[0]
            formatted = f'Valid: {item.valid}, {name}: [bold]{item.avg * 100:.2f}[/bold]'
            if item.std is not None:
                formatted += f' ± {item.std * 100:.2f}'
            if item.bon is not None:
                formatted += f', BoN: {item.bon * 100:.2f}'

            self._result_summary_for_metric = True
            self._result_summary.update(formatted)

        self._tabs.hide_tab('tab_metrics')

    async def _handle_token_usage_summary(self, e: TokenUsageSummaryEvent):
        self._token_summary.update(e.summary)

    async def _handle_result_list(self, e: ResultListEvent):
        self._tabs.show_tab('tab_resume')
        self._tabs.active = 'tab_resume'
        self._result_summary.update('Waiting for Input...')
        self._resume_list.clear_options()
        self._resume_list.add_options([
            Option('Start New Evaluation', None),
            None,  # divider
            *[Option(result.name, str(result.path)) for result in e.items]
        ])
        self._resume_list.highlighted = 0
        self._resume_list.focus()

    async def _handle_specs(self, e: SpecsEvent):
        self._result_summary.update('Running Evaluation...')
        self._progress_bar.update(total=len(e.items), progress=0)

        table = self._sessions_table
        table.clear()
        table.add_columns(
            ('Model', 'model'),
            ('Task', 'task'),
            ('Index', 'index'),
            ('Run', 'run'),
            ('Session ID', 'session_id'),
            ('Status', 'status'),
            ('Time Used', 'time_used')
        )
        for spec in e.items:
            table.add_row(
                spec.model,
                spec.task,
                spec.index,
                spec.run,
                '',
                Text('○ Pending', style='grey50'),
                '',
                height=1,
                key=spec.run_key()
            )
        table.sort('run', 'task', 'index', 'model')

        self._tabs.show_tab('tab_sessions')
        self._tabs.active = 'tab_sessions'

    async def _handle_session_started(self, e: SessionStartedEvent):
        table = self._sessions_table
        try:
            table.update_cell(
                e.spec.run_key(),
                'session_id',
                e.session_id
            )
            table.update_cell(
                e.spec.run_key(),
                'status',
                Text('▹ In Progress', style='cornflower_blue'),
                update_width=True
            )
        except CellDoesNotExist:
            pass

    async def _handle_session_completed(self, e: SessionCompletedEvent):
        self._progress_bar.update(advance=1)

        table = self._sessions_table
        try:
            table.update_cell(
                e.spec.run_key(),
                'session_id',
                e.session_id
            )

            value, metric = e.metric
            if metric == MetricResult.SUCCESS:
                status_text = Text(f'✔ {e.result.status}: {value}', style='bold green3')
            elif metric == MetricResult.PARTIAL_SUCCESS:
                status_text = Text(f'≈ {e.result.status}: {value}', style='bold dark_orange3')
            elif metric == MetricResult.FAILURE:
                status_text = Text(f'✘ {e.result.status}: {value}', style='bold indian_red')
            elif metric == MetricResult.UNKNOWN:
                status_text = Text(f'⊙ {e.result.status}: {value}', style='bold cyan')
            else:
                status_text = Text(f'‼ {e.result.status}', style='bold bright_red')
            table.update_cell(
                e.spec.run_key(),
                'status',
                status_text,
                update_width=True
            )

            if e.result.time_start is not None and e.result.time_end is not None:
                time_used = e.result.time_end - e.result.time_start
                table.update_cell(
                    e.spec.run_key(),
                    'time_used',
                    f'{time_used.total_seconds():.1f}s'
                )
        except CellDoesNotExist:
            pass

    async def _handle_workflow_completed(self, _: WorkflowCompletedEvent):
        if not self._result_summary_for_metric:
            self._result_summary.update('Completed.')
        self._progress_bar.update(progress=self._progress_bar.total)

        table = self._sessions_table
        for key in table.rows:
            try:
                status = table.get_cell(key, 'status')
                if isinstance(status, Text) and 'In Progress' in status.plain:
                    table.update_cell(
                        key,
                        'status',
                        Text('‼ cancelled', style='bold bright_red'),
                        update_width=True
                    )
            except CellDoesNotExist:
                pass

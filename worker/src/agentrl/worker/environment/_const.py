import re

LABEL_MANAGED_BY = 'agentrl.managed-by'
LABEL_MANAGED_BY_VALUE = 'agentrl'
LABEL_TASK_NAME = 'agentrl.task-name'
LABEL_SUBTYPE_NAME = 'agentrl.subtype-name'
LABEL_DEPENDS_ON = 'agentrl.depends-on'
LABEL_EXCLUSIVE = 'agentrl.exclusive'

K8S_LABEL_PREFIX = 'agentrl.live/'
K8S_LABEL_ROLE = f'{K8S_LABEL_PREFIX}role'
k8S_LABEL_ROLE_ENVIRONMENT = 'environment'
K8S_LABEL_TASK_TYPE = f'{K8S_LABEL_PREFIX}task-type'
K8S_LABEL_SUBTYPE_NAME = f'{K8S_LABEL_PREFIX}subtype-name'
K8S_LABEL_EXCLUSIVE = f'{K8S_LABEL_PREFIX}exclusive'
K8S_LABEL_KVM = f'{K8S_LABEL_PREFIX}kvm'
K8S_LABEL_BTRFS = f'{K8S_LABEL_PREFIX}btrfs'
K8S_ANNOTATION_SESSION_ID = f'{K8S_LABEL_PREFIX}session-id'
K8S_ANNOTATION_DEPENDS_ON = f'{K8S_LABEL_PREFIX}depends-on'

SHELL_PROMPT_RE = re.compile(b'\x1b.+@.+[#|$] ')

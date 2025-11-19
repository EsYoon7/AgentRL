package utils

import "go.uber.org/zap"

func NewLogger(debug bool) *zap.SugaredLogger {
	var logger *zap.Logger

	if debug {
		logger, _ = zap.NewDevelopment()
	} else {
		logger, _ = zap.NewProduction()
	}

	return logger.Sugar()
}

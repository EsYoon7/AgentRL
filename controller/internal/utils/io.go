package utils

type CountWriter struct {
	n uint64
}

func NewCountWriter() *CountWriter {
	return &CountWriter{}
}

func (w *CountWriter) Write(p []byte) (int, error) {
	w.n += uint64(len(p))
	return len(p), nil
}

func (w *CountWriter) Count() uint64 {
	return w.n
}

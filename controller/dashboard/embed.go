package dashboard

import (
	"embed"
	"io/fs"
	"log"
	"net/http"
)

//go:embed dist/**
var staticFiles embed.FS

func FS() http.FileSystem {
	distFS, err := fs.Sub(staticFiles, "dist")
	if err != nil {
		log.Fatalf("Failed to load assets: %v", err)
	}
	return http.FS(distFS)
}

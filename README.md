# go-tfidf
[![Build Status](https://travis-ci.org/NeowayLabs/go-tfidf.svg?branch=master)](https://travis-ci.org/NeowayLabs/go-tfidf)
[![Go Report Card](https://goreportcard.com/badge/github.com/NeowayLabs/go-tfidf)](https://goreportcard.com/report/github.com/NeowayLabs/go-tfidf)

This project implements a library that computes Tf Idf for text documents and similarity


# Requirements

- Go 1.16

# Running tests

## Requirements

- Docker

Run the following command:

```
make check
```

# Install

In your Go project directory, run the following command:

```
go get -u github.com/NeowayLabs/go-tfidf
```

# Example

```go
import (
	go_tfidf "github.com/NeowayLabs/go-tfidf"
	"github.com/NeowayLabs/go-tfidf/similarity"
)

func main() {
	separator := " "
	similarityFunction := "Cosine"

	inputDocuments := []string{
			"The game of life is a game of everlasting learning",
			"The unexamined life is not worth living",
			"Never stop learning",
		}
	inputQuery := "life learning"

	ti, err := go_tfidf.New(inputDocuments, separator)

	queryTfIdfDocuments, err := ti.CalculateQueryTermsTfIdfForEachDocument(inputQuery)
	queryTfIdf, err := go_tfidf.CalculateQueryTermsTfIdf(inputQuery, ti.DocumentSeparator)

	similarities, err := similarity.CalculateSimilarities(queryTfIdf, queryTfIdfDocuments, similarityFunction)
}

```


References: https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
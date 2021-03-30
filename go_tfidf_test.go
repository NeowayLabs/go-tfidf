package go_tfidf_test

import (
	"testing"

	"github.com/NeowayLabs/go-tfidf/similarity"

	go_tfidf "github.com/NeowayLabs/go-tfidf"
	"github.com/stretchr/testify/assert"
)

func TestSetSeparator(t *testing.T) {
	ti := go_tfidf.New()

	newSeparator := "-"
	expected := "-"
	ti.SetSeparator(newSeparator)

	assert.Equal(t, expected, ti.DocumentSeparator)
}

// The main reference for implementing this lib was https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
// Thus, the tests that are going to be used are based on the tutorial example
func TestTfIdfWithCosineSimilarity(t *testing.T) {
	inputDocuments := []string{
		"The game of life is a game of everlasting learning",
		"The unexamined life is not worth living",
		"Never stop learning",
	}
	inputQuery := "life learning"

	expectedNormalizedTf := []map[string]float64{
		map[string]float64{
			"the":         0.1,
			"game":        0.2,
			"of":          0.2,
			"life":        0.1,
			"is":          0.1,
			"a":           0.1,
			"everlasting": 0.1,
			"learning":    0.1,
		},
		map[string]float64{
			"the":        0.14285714285714285,
			"unexamined": 0.14285714285714285,
			"life":       0.14285714285714285,
			"is":         0.14285714285714285,
			"not":        0.14285714285714285,
			"worth":      0.14285714285714285,
			"living":     0.14285714285714285,
		},
		map[string]float64{
			"never":    0.3333333333333333,
			"stop":     0.3333333333333333,
			"learning": 0.3333333333333333,
		},
	}
	expectedDocumentTerms := []string{
		"the",
		"game",
		"of",
		"life",
		"is",
		"a",
		"everlasting",
		"learning",
		"unexamined",
		"not",
		"worth",
		"living",
		"never",
		"stop",
	}
	expectedDocumentIdf := map[string]float64{
		"the":         1.4054651081081644,
		"game":        2.0986122886681096,
		"of":          2.0986122886681096,
		"life":        1.4054651081081644,
		"is":          1.4054651081081644,
		"a":           2.0986122886681096,
		"everlasting": 2.0986122886681096,
		"learning":    1.4054651081081644,
		"unexamined":  2.0986122886681096,
		"not":         2.0986122886681096,
		"worth":       2.0986122886681096,
		"living":      2.0986122886681096,
		"never":       2.0986122886681096,
		"stop":        2.0986122886681096,
	}
	expectedSimilarities := []float64{
		1.0,
		0.7071067811865475,
		0.7071067811865475,
	}

	ti := go_tfidf.New()
	ti.AddDocuments(inputDocuments)

	assert.Equal(t, expectedNormalizedTf, ti.DocumentsNormTermFrequency)
	assert.Equal(t, expectedDocumentTerms, ti.DocumentsTerms)

	ti.CalculateDocumentsIdf()

	assert.Equal(t, expectedDocumentIdf, ti.DocumentsInverseFrequency)

	queryTfIdfDocuments, err := ti.CalculateQueryTfIdfForEveryDocument(inputQuery)
	assert.Nil(t, err)

	queryTfIdf := go_tfidf.CalculateQueryTfIdf(inputQuery, ti.DocumentSeparator)

	similarities, err := similarity.CalculateSimilarities(queryTfIdf, queryTfIdfDocuments, "Cosine")

	assert.Equal(t, expectedSimilarities, similarities)
	assert.Nil(t, err)
}

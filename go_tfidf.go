package go_tfidf

import (
	"errors"
	"math"
	"strings"

	"github.com/NeowayLabs/go-tfidf/helper"
)

type TfIdf struct {
	DocumentSeparator          string
	Documents                  []string
	DocumentsNormTermFrequency []map[string]float64
	DocumentsTerms             []string
	DocumentsInverseFrequency  map[string]float64
}

func (ti *TfIdf) AddDocuments(documents []string) error {
	if len(documents) < 1 {
		return errors.New("At least one document must be passed!")
	}

	for _, doc := range documents {
		ti.Documents = append(ti.Documents, doc)
		docTerms := strings.Split(strings.ToLower(doc), ti.DocumentSeparator)
		ti.DocumentsTerms = append(ti.DocumentsTerms, docTerms...)

		ti.DocumentsNormTermFrequency = append(ti.DocumentsNormTermFrequency, normalizedTermFrequency(docTerms))
	}
	ti.DocumentsTerms = helper.RemoveDuplicates(ti.DocumentsTerms)

	return nil
}

func normalizedTermFrequency(terms []string) map[string]float64 {
	normalizedTermFrequencies := make(map[string]float64, 0)

	nTerms := float64(len(terms))
	for _, term := range terms {
		normalizedTermFrequencies[term] += 1.0 / nTerms
	}

	return normalizedTermFrequencies
}

func (ti *TfIdf) CalculateDocumentsIdf() {
	for _, term := range ti.DocumentsTerms {
		ti.DocumentsInverseFrequency[term] = inverseDocumentFrequency(term, ti.Documents, ti.DocumentSeparator)
	}
}

func inverseDocumentFrequency(term string, documents []string, separator string) float64 {
	countTermsInDocuments := 0
	for _, doc := range documents {
		docTerms := strings.Split(strings.ToLower(doc), separator)
		if helper.StringArrayContainsWord(docTerms, strings.ToLower(term)) {
			countTermsInDocuments++
		}
	}

	if countTermsInDocuments > 0 {
		return 1.0 + math.Log(float64(len(documents))/float64(countTermsInDocuments))
	}

	return 1.0

}

func (ti *TfIdf) CalculateQueryTfIdfForEveryDocument(query string) ([][]float64, error) {
	queryTerms := strings.Split(query, ti.DocumentSeparator)
	termsTfIdfs := make([][]float64, 0)

	if len(queryTerms) == 1 && queryTerms[0] == "" {
		return termsTfIdfs, errors.New("Query must have at least one term")
	}

	for docIdx, docNormTf := range ti.DocumentsNormTermFrequency {
		termsTfIdfs = append(termsTfIdfs, make([]float64, 0))
		for _, term := range queryTerms {
			tf := 0.0
			idf := 0.0
			if v, ok := docNormTf[term]; ok {
				tf = v
			}
			if v, ok := ti.DocumentsInverseFrequency[term]; ok {
				idf = v
			}
			termsTfIdfs[docIdx] = append(termsTfIdfs[docIdx], tf*idf)
		}
	}

	return termsTfIdfs, nil
}

func CalculateQueryTfIdf(query string, separator string) ([]float64, error) {
	docs := []string{query}
	queryTerms := strings.Split(query, separator)
	queryTfIdf := make([]float64, 0)

	if len(queryTerms) == 1 && queryTerms[0] == "" {
		return queryTfIdf, errors.New("Query must have at least one term")
	}

	termFrequencies := normalizedTermFrequency(queryTerms)

	for _, term := range queryTerms {
		tf := termFrequencies[term]
		idf := inverseDocumentFrequency(term, docs, separator)
		queryTfIdf = append(queryTfIdf, tf*idf)
	}

	return queryTfIdf, nil
}

func (ti *TfIdf) SetSeparator(sep string) {
	ti.DocumentSeparator = sep
}

func New() *TfIdf {
	return &TfIdf{
		DocumentSeparator:          " ",
		Documents:                  make([]string, 0),
		DocumentsNormTermFrequency: make([]map[string]float64, 0),
		DocumentsTerms:             make([]string, 0),
		DocumentsInverseFrequency:  make(map[string]float64, 0),
	}
}

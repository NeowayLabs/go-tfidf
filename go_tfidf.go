package go_tfidf

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/NeowayLabs/go-tfidf/helper"
)

type TfIdf struct {
	documentSeparator          string
	documents                  []string
	documentsNormTermFrequency []map[string]float64
	documentsTerms             []string
	documentsInverseFrequency  map[string]float64
}

func (ti *TfIdf) AddDocuments(documents []string) error {
	if len(documents) < 1 {
		return errors.New("At least one document must be passed!")
	}

	for _, doc := range documents {
		docTerms := strings.Split(strings.ToLower(doc), ti.documentSeparator)
		if len(docTerms) < 1 || (len(docTerms) == 1 && docTerms[0] == "") {
			ti.documents = make([]string, 0)
			return errors.New(fmt.Sprintf("Document error. %s document is invalid", doc))
		}
		ti.documents = append(ti.documents, doc)
		ti.documentsTerms = append(ti.documentsTerms, docTerms...)

		ti.documentsNormTermFrequency = append(ti.documentsNormTermFrequency, normalizedTermFrequency(docTerms))
	}
	ti.documentsTerms = helper.RemoveDuplicates(ti.documentsTerms)

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
	for _, term := range ti.documentsTerms {
		ti.documentsInverseFrequency[term] = inverseDocumentFrequency(term, ti.documents, ti.documentSeparator)
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

func (ti *TfIdf) CalculateQueryTermsTfIdfForEachDocument(query string) ([][]float64, error) {
	queryTerms := strings.Split(query, ti.documentSeparator)
	termsTfIdfs := make([][]float64, 0)

	if len(queryTerms) == 1 && queryTerms[0] == "" {
		return termsTfIdfs, errors.New("Query must have at least one term")
	}

	for docIdx, docNormTf := range ti.documentsNormTermFrequency {
		termsTfIdfs = append(termsTfIdfs, make([]float64, 0))
		for _, term := range queryTerms {
			tf := 0.0
			idf := 0.0
			if v, ok := docNormTf[term]; ok {
				tf = v
			}
			if v, ok := ti.documentsInverseFrequency[term]; ok {
				idf = v
			}
			termsTfIdfs[docIdx] = append(termsTfIdfs[docIdx], tf*idf)
		}
	}

	return termsTfIdfs, nil
}

func CalculateQueryTermsTfIdf(query string, separator string) ([]float64, error) {
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

func (ti *TfIdf) SetDocumentSeparator(separator string) {
	ti.documentSeparator = separator
}

func (ti *TfIdf) GetDocumentSeparator() string {
	return ti.documentSeparator
}

func (ti *TfIdf) GetDocuments() []string {
	return ti.documents
}

func (ti *TfIdf) GetDocumentsNormTermFrequency() []map[string]float64 {
	return ti.documentsNormTermFrequency
}

func (ti *TfIdf) GetDocumentsInverseFrequency() map[string]float64 {
	return ti.documentsInverseFrequency
}

func (ti *TfIdf) GetDocumentsTerms() []string {
	return ti.documentsTerms
}

func New() *TfIdf {
	return &TfIdf{
		documentSeparator:          " ",
		documents:                  make([]string, 0),
		documentsNormTermFrequency: make([]map[string]float64, 0),
		documentsTerms:             make([]string, 0),
		documentsInverseFrequency:  make(map[string]float64, 0),
	}
}

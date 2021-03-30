package go_tfidf

import (
	"math"
	"strings"
)

type TfIdf struct {
	DocumentSeparator          string
	Documents                  []string
	DocumentsNormTermFrequency []map[string]float64
	DocumentsTerms             []string
	DocumentsInverseFrequency  map[string]float64
}

func (ti *TfIdf) AddDocuments(documents []string) {
	for _, doc := range documents {
		ti.Documents = append(ti.Documents, doc)
		docTerms := strings.Split(strings.ToLower(doc), ti.DocumentSeparator)
		ti.DocumentsTerms = append(ti.DocumentsTerms, docTerms...)

		ti.DocumentsNormTermFrequency = append(ti.DocumentsNormTermFrequency, normalizedTermFrequency(docTerms))
	}
}

func normalizedTermFrequency(terms []string) map[string]float64 {
	normalizedTermFrequencies := make(map[string]float64, 0)

	nTerms := len(terms)
	for _, term := range terms {
		normalizedTermFrequencies[term] += 1.0 / float64(nTerms)
	}

	return normalizedTermFrequencies
}

func (ti *TfIdf) CalculateDocumentsIdf() {
	for _, term := range ti.DocumentsTerms {
		ti.DocumentsInverseFrequency[term] = inverseDocumentFrequency(term, ti.Documents)
	}
}

func inverseDocumentFrequency(term string, documents []string) float64 {
	countTermsInDocuments := 0
	for _, doc := range documents {
		if strings.Contains(strings.ToLower(doc), strings.ToLower(term)) {
			countTermsInDocuments++
		}
	}

	if countTermsInDocuments > 0 {
		return 1.0 + math.Log(float64(len(documents)/countTermsInDocuments))
	}

	return 1.0

}

func (ti *TfIdf) CalculateQueryTfIdfForEveryDocument(query string) map[int][]float64 {
	queryTerms := strings.Split(query, ti.DocumentSeparator)
	termsTfIdfs := make(map[int][]float64, 0)

	if len(queryTerms) < 1 {
		return termsTfIdfs
	}

	for docIdx, docNormTf := range ti.DocumentsNormTermFrequency {
		termsTfIdfs[docIdx] = make([]float64, 0)
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

	return termsTfIdfs
}

func (ti *TfIdf) CalculateQueryTfIdf(query string) []float64 {
	docs := []string{query}
	queryTerms := strings.Split(query, ti.DocumentSeparator)
	queryTfIdf := make([]float64, 0)

	if len(queryTerms) < 1 {
		return queryTfIdf
	}

	termFrequencies := normalizedTermFrequency(queryTerms)

	for _, term := range queryTerms {
		tf := termFrequencies[term]
		idf := inverseDocumentFrequency(term, docs)
		queryTfIdf = append(queryTfIdf, tf*idf)
	}

	return queryTfIdf
}

func (ti *TfIdf) SetSeparator(sep string) {
	ti.DocumentSeparator = sep
}

// Criar um new ti
// Adicionar documentos
// Calcular idf dos documentos
// Funcao que dada uma query, retorna o tf*idf de cada termo da query para todos os documentos [][]float64
// Funcao que calcula o tf*idf dos termos da query

// Entao, calcular a similaridade entre a query e os documentos

func New() *TfIdf {
	return &TfIdf{
		DocumentSeparator:          " ",
		Documents:                  make([]string, 0),
		DocumentsNormTermFrequency: make([]map[string]float64, 0),
		DocumentsTerms:             make([]string, 0),
		DocumentsInverseFrequency:  make(map[string]float64, 0),
	}
}

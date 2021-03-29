package tfidf

import (
	"math"
	"strings"
)

type TfIdf struct {
	DocumentSeparator          string
	Documents                  []string
	DocumentsNormTermFrequency []map[string]float64
	DocumentsTerms             map[string]int
	DocumentsInverseFrequency  map[string]float64
}

func (ti *TfIdf) AddDocuments(documents []string) {
	ti.Documents = append(ti.Documents, documents)

	for idx, doc := range documents {
		docTerms := strings.Split(strings.ToLower(doc), ti.DocumentSeparator)
		nTerms := len(docTerms)
		ti.DocumentsNormTermFrequency = append(ti.DocumentsNormTermFrequency, make(map[string]float64, 0))

		for _, docTerm := range docTerms {
			if _, ok := ti.DocumentsTermFrequency[docTerm].(int); !ok {
				ti.DocumentsTerms[docTerm] = 1
			} else {
				ti.DocumentsTerms[docTerm]++
			}
			ti.DocumentsNormTermFrequency[idx][docTerm] += 1.0 / nTerms
		}
	}

}

func (ti *TfIdf) CalculateDocumentsIdf() {
	for term, _ := range ti.DocumentsTerms {
		DocumentsInverseFrequency[term] := idf(term, ti.Documents)
	}
}

func idf(term string, documents []string) float64 {
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
		DocumentsTerms:             make(map[string]int, 0),
		DocumentsInverseFrequency:  make(map[string]float64, 0),
	}
}

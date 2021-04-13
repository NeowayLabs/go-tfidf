// Package go_tfidf provides a Tf-Idf implementation.
package go_tfidf

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// A TfIdf represents the set of variables that are used for computing the reference documents Tf and Idf values.
type TfIdf struct {
	// DocumentSeparator is the string that is going to be used to split the documents terms.
	DocumentSeparator string
	// documents are the set of reference documents that are going to be used to compare with input queries.
	documents []string
	// documentsNormTermFrequency are the normalized term frequencies for all TfIdf.documents (Tf).
	documentsNormTermFrequency []map[string]float64
	// documentsTerms are the terms for all TfIdf.documents splitted by the TfIdf.DocumentSeparator.
	documentsTerms []string
	// documentsInverseFrequency are the TfIdf.documentTerms Inverse Document Frequency (Idf).
	documentsInverseFrequency map[string]float64
}

// AddDocuments receives an array of strings containing the documents that are going to be used as references.
// If the array is empty or any of the input documents are invalid, it return an error.
// An invalid document has no terms or only has one, but is an empty string.
func (ti *TfIdf) AddDocuments(documents []string) error {
	if len(documents) < 1 {
		return errors.New("at least one document must be passed")
	}

	for _, doc := range documents {
		docTerms := strings.Split(strings.ToLower(doc), ti.DocumentSeparator)
		if len(docTerms) < 1 || (len(docTerms) == 1 && docTerms[0] == "") {
			ti.documents = make([]string, 0)
			return fmt.Errorf("document error. %s document is invalid", doc)
		}
		ti.documents = append(ti.documents, doc)
		ti.documentsTerms = append(ti.documentsTerms, docTerms...)

		ti.documentsNormTermFrequency = append(ti.documentsNormTermFrequency, normalizedTermFrequency(docTerms))
	}

	ti.documentsTerms = removeDuplicates(ti.documentsTerms)
	ti.calculateDocumentsIdf()

	return nil
}

// CalculateQueryTermsTfIdfForEachDocument receives a query string and computes Tf-Idf of its terms for every document in the *TfIdf object.
// If the query term is an empty string, returns an error.
func (ti *TfIdf) CalculateQueryTermsTfIdfForEachDocument(query string) ([][]float64, error) {
	queryTerms := strings.Split(strings.ToLower(query), ti.DocumentSeparator)
	termsTfIdfs := make([][]float64, 0)

	if len(queryTerms) == 1 && queryTerms[0] == "" {
		return termsTfIdfs, errors.New("query must have at least one term")
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

// CalculateQueryTermsTfIdf receives a query string with a separator (*TfIdf.DocumentSeparator) and computes the TfIdfs value for each term.
// If the query term is an empty string, returns an error.
func CalculateQueryTermsTfIdf(query string, separator string) ([]float64, error) {
	docs := []string{query}
	queryTerms := strings.Split(strings.ToLower(query), separator)
	queryTfIdf := make([]float64, 0)

	if len(queryTerms) == 1 && queryTerms[0] == "" {
		return queryTfIdf, errors.New("query must have at least one term")
	}

	termFrequencies := normalizedTermFrequency(queryTerms)

	for _, term := range queryTerms {
		tf := termFrequencies[term]
		idf := inverseDocumentFrequency(term, docs, separator)
		queryTfIdf = append(queryTfIdf, tf*idf)
	}

	return queryTfIdf, nil
}

func removeDuplicates(words []string) []string {
	uniqueWords := make([]string, 0)

	keys := make(map[string]bool)
	for _, w := range words {
		if _, exists := keys[w]; !exists {
			uniqueWords = append(uniqueWords, w)
			keys[w] = true
		}
	}

	return uniqueWords
}

func stringArrayContainsWord(words []string, word string) bool {
	for _, w := range words {
		if word == w {
			return true
		}
	}

	return false
}

// Documents returns the *TfIdf.documents private attribute values.
func (ti *TfIdf) Documents() []string {
	return ti.documents
}

// DocumentsNormTermFrequency returns the *TfIdf.documentsNormTermFrequency private attribute values.
func (ti *TfIdf) DocumentsNormTermFrequency() []map[string]float64 {
	return ti.documentsNormTermFrequency
}

// DocumentsInverseFrequency returns the *TfIdf.DocumentsInverseFrequency private attribute values.
func (ti *TfIdf) DocumentsInverseFrequency() map[string]float64 {
	return ti.documentsInverseFrequency
}

// DocumentsInverseFrequency returns the *TfIdf.documentsTerms private attribute values.
func (ti *TfIdf) DocumentsTerms() []string {
	return ti.documentsTerms
}

func normalizedTermFrequency(terms []string) map[string]float64 {
	normalizedTermFrequencies := make(map[string]float64, 0)
	nTerms := float64(len(terms))

	for _, term := range terms {
		normalizedTermFrequencies[term] += 1.0 / nTerms
	}

	return normalizedTermFrequencies
}

func (ti *TfIdf) calculateDocumentsIdf() {
	for _, term := range ti.documentsTerms {
		ti.documentsInverseFrequency[term] = inverseDocumentFrequency(term, ti.documents, ti.DocumentSeparator)
	}
}

func inverseDocumentFrequency(term string, documents []string, separator string) float64 {
	countTermsInDocuments := 0

	for _, doc := range documents {
		docTerms := strings.Split(strings.ToLower(doc), separator)
		if stringArrayContainsWord(docTerms, strings.ToLower(term)) {
			countTermsInDocuments++
		}
	}

	if countTermsInDocuments > 0 {
		return 1.0 + math.Log(float64(len(documents))/float64(countTermsInDocuments))
	}

	return 1.0

}

func New(documents []string, separator string) (*TfIdf, error) {
	ti := TfIdf{
		DocumentSeparator:          separator,
		documents:                  make([]string, 0),
		documentsNormTermFrequency: make([]map[string]float64, 0),
		documentsTerms:             make([]string, 0),
		documentsInverseFrequency:  make(map[string]float64, 0),
	}
	err := ti.AddDocuments(documents)
	if err != nil {
		return nil, err
	}

	return &ti, nil
}

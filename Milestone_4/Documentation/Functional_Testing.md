# Functional Testing Report
## AI-Powered TED Talk Recommendation System

### 1. Introduction

Functional testing was performed to verify the major features and operations of the TED Talk Recommendation System.

The objective was to ensure that the Streamlit application loads correctly, accepts user selections, generates recommendations, applies filters, sorts results, and displays relevant TED Talk information.

### 2. Testing Environment

| Component | Configuration |
|---|---|
| Operating System | Windows |
| Programming Language | Python 3.11 |
| Development Environment | Visual Studio Code |
| Application Framework | Streamlit |
| Machine Learning Library | Scikit-learn |
| Recommendation Technique | Content-Based Filtering |
| Text Vectorization | TF-IDF |
| Similarity Measure | Cosine Similarity |

### 3. Functional Test Cases

| Test ID | Test Scenario | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC01 | Launch the Streamlit application | Application should open successfully | To be recorded | Pending |
| TC02 | Load the prepared TED Talks dataset | Dataset should load without errors | To be recorded | Pending |
| TC03 | Display application statistics | Total talks, speakers, and topics should be displayed | To be recorded | Pending |
| TC04 | Select a TED Talk | Selected talk should be displayed | To be recorded | Pending |
| TC05 | Generate 5 recommendations | Up to 5 recommendations should be generated | To be recorded | Pending |
| TC06 | Generate 10 recommendations | Up to 10 recommendations should be generated | To be recorded | Pending |
| TC07 | Apply topic filtering | Results should match the selected topic | To be recorded | Pending |
| TC08 | Sort by similarity score | Recommendations should be ordered by similarity score | To be recorded | Pending |
| TC09 | Sort by views | Recommendations should be ordered by view count | To be recorded | Pending |
| TC10 | Sort by duration | Recommendations should be ordered by duration | To be recorded | Pending |
| TC11 | Display recommendation details | Title, speaker, topics, and description should appear | To be recorded | Pending |
| TC12 | Open TED Talk URL | Corresponding TED Talk should open | To be recorded | Pending |

### 4. Testing Procedure

The following procedure was used to test the application:

1. Launch the application using the Streamlit command.
2. Verify that the prepared dataset and trained vectorizer load successfully.
3. Check the statistics displayed on the home page.
4. Select a TED Talk using the dropdown.
5. Generate recommendations using both 5 and 10 recommendation options.
6. Apply different topic filters.
7. Test the available sorting options.
8. Verify the information displayed for recommended talks.
9. Check whether the TED Talk links work correctly.

### 5. Testing Results

Record the actual result and update the status of each test case after executing the corresponding test.

- **PASS:** The feature works as expected.
- **FAIL:** The feature does not work as expected.
- **PENDING:** Testing has not yet been completed.

### 6. Conclusion

Functional testing is intended to verify the reliability and correctness of the TED Talk Recommendation System.

The test cases cover the main application features, including dataset loading, TED Talk selection, recommendation generation, topic filtering, sorting, and recommendation display.

The final testing status should be updated after all test cases have been executed.
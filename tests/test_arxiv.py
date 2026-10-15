import io
import os
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import patch

from agentic_paper_retrieval.sources.arxiv import ArxivSource, keyword_query, parse_feed
from agentic_paper_retrieval.sources.base import SourceError

FEED = (Path(__file__).parent / "fixtures/arxiv.xml").read_bytes()
EMPTY = b'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/"><o:totalResults>0</o:totalResults></feed>'


class ArxivTests(unittest.TestCase):
    def test_question_to_keywords_and_quoted_phrases(self):
        self.assertEqual(keyword_query('Find papers about "medical QA" and RAG'), 'all:"medical QA" AND all:"RAG"')
        self.assertEqual(keyword_query("C++ retrieval"), 'all:"C++" AND all:"retrieval"')
        with self.assertRaises(ValueError):
            keyword_query("how can the")

    def test_normalization_and_version_deduplication(self):
        result = parse_feed(FEED, "retrieval", 'all:"retrieval"')
        self.assertEqual(result.total_results, 3)
        self.assertEqual(len(result.papers), 2)
        paper = result.papers[0]
        self.assertEqual(paper.id, "arxiv:2501.99999")
        self.assertEqual(paper.source_id, "2501.99999v2")
        self.assertEqual(paper.title, "Synthetic Retrieval Study")
        self.assertEqual(paper.authors, ("Example Researcher", "Second Author"))
        self.assertEqual(paper.doi, "10.0000/synthetic")
        self.assertEqual(paper.categories, ("cs.CL",))
        self.assertTrue(paper.pdf_url.startswith("https://arxiv.org/pdf/"))
        self.assertEqual(result.papers[1].id, "arxiv:cs/9909999")

    def test_empty_results_are_valid_but_bad_responses_are_not(self):
        self.assertEqual(parse_feed(EMPTY, "q", "q").papers, ())
        for data in (b"not XML", b"<html/>", b'<feed xmlns="http://www.w3.org/2005/Atom"/>'):
            with self.subTest(data=data), self.assertRaises(SourceError):
                parse_feed(data, "q", "q")
        error_feed = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#incorrect_id_format</id><summary>Invalid query</summary></entry></feed>'
        with self.assertRaisesRegex(SourceError, "Invalid query"):
            parse_feed(error_feed, "q", "q")

    def test_cache_and_expiry(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ArxivSource(Path(directory))
            with patch.object(source, "_fetch", return_value=FEED) as fetch:
                first = source.search("retrieval", 3)
                self.assertEqual(source.search("retrieval", 3), first)
                self.assertEqual(fetch.call_count, 1)
                cache_file = next(Path(directory).glob("*.xml"))
                os.utime(cache_file, (0, 0))
                source.search("retrieval", 3)
                self.assertEqual(fetch.call_count, 2)

    def test_raw_query_and_url_encoding(self):
        source = ArxivSource(cache_dir=None)
        with patch.object(source, "_fetch", return_value=EMPTY) as fetch:
            query = 'cat:cs.CL AND ti:"retrieval augmented"'
            source.search(query, 7, raw_query=True)
            url = fetch.call_args.args[0]
            params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            self.assertEqual(params["search_query"], [query])
            self.assertEqual(params["max_results"], ["7"])
        for query, limit in (("", 3), ("q", 0), ("q", 51)):
            with self.assertRaises(ValueError):
                source.search(query, limit)

    def test_network_error_is_actionable_and_requests_are_paced(self):
        source = ArxivSource(cache_dir=None)
        with patch("agentic_paper_retrieval.sources.arxiv._last_request", 100), \
             patch("agentic_paper_retrieval.sources.arxiv.time.monotonic", return_value=101), \
             patch("agentic_paper_retrieval.sources.arxiv.time.sleep") as sleep, \
             patch("agentic_paper_retrieval.sources.arxiv.urllib.request.urlopen", return_value=io.BytesIO(EMPTY)):
            source.search("retrieval")
            sleep.assert_called_once_with(2)
        with patch("agentic_paper_retrieval.sources.arxiv._last_request", None), \
             patch("agentic_paper_retrieval.sources.arxiv.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaisesRegex(SourceError, "Could not reach arXiv"):
                source.search("retrieval")

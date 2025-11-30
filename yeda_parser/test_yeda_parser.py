"""Tests for yeda_parser.py."""

import unittest
from yeda_parser import parse_yeda_string


class TestYedaParser(unittest.TestCase):
    """Test YEDDA parsing functions."""

    def test_parse_simple_block(self):
        """Test parsing a simple YEDDA block."""
        text = "[@ Line 1\nLine 2\n#Label*]"
        result = parse_yeda_string(text)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ('Label', 'Line 1', 0))
        self.assertEqual(result[1], ('Label', 'Line 2', 1))

    def test_parse_multiple_blocks(self):
        """Test parsing multiple YEDDA blocks."""
        text = """[@ First block line 1
First block line 2
#Label1*]
[@ Second block line 1
#Label2*]"""

        result = parse_yeda_string(text)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], 'Label1')
        self.assertEqual(result[1][0], 'Label1')
        self.assertEqual(result[2][0], 'Label2')

    def test_parse_with_whitespace(self):
        """Test parsing with extra whitespace."""
        text = "[@   Line with spaces   \n  Another line  \n#Label*]"
        result = parse_yeda_string(text)

        self.assertEqual(len(result), 2)
        # Leading/trailing whitespace around content block is stripped by regex
        # but internal line whitespace is preserved
        self.assertEqual(result[0][1], 'Line with spaces   ')
        self.assertEqual(result[1][1], '  Another line')

    def test_parse_empty_lines(self):
        """Test parsing with empty lines in content."""
        text = "[@ Line 1\n\nLine 3\n#Label*]"
        result = parse_yeda_string(text)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][1], 'Line 1')
        self.assertEqual(result[1][1], '')
        self.assertEqual(result[2][1], 'Line 3')

    def test_parse_nomenclature_label(self):
        """Test parsing with Nomenclature label."""
        text = "[@ Glomus mosseae Nicolson & Gerdemann, 1963.\n#Nomenclature*]"
        result = parse_yeda_string(text)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 'Nomenclature')
        self.assertTrue('Glomus mosseae' in result[0][1])

    def test_parse_description_label(self):
        """Test parsing with Description label."""
        text = "[@ Type species: Glomus mosseae\nSpores: 100-200 µm\n#Description*]"
        result = parse_yeda_string(text)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 'Description')
        self.assertEqual(result[1][0], 'Description')

    def test_parse_multiline_content(self):
        """Test parsing with multiline content."""
        text = """[@ This is a long paragraph
that spans multiple lines
and contains various information
about taxonomy.
#Misc-exposition*]"""

        result = parse_yeda_string(text)

        self.assertEqual(len(result), 4)
        for label, line, line_num in result:
            self.assertEqual(label, 'Misc-exposition')
            self.assertEqual(line_num, result.index((label, line, line_num)))

    def test_line_numbers(self):
        """Test that line numbers are correctly assigned."""
        text = "[@ Line 0\nLine 1\nLine 2\n#Label*]"
        result = parse_yeda_string(text)

        for i, (label, line, line_num) in enumerate(result):
            self.assertEqual(line_num, i)
            self.assertEqual(line, f'Line {i}')

    def test_parse_real_world_example(self):
        """Test parsing a real-world YEDDA example."""
        text = """[@ ISSN (print) 0093-4666
© 2011. Mycotaxon, Ltd.
#Misc-exposition*]
[@ Simiglomus hoi (S.M. Berch & Trappe) G.A. Silva, Oehl & Sieverd., comb. nov.
MycoBank MB 518461
≡ Glomus hoi S.M. Berch & Trappe, Mycologia 77: 654. 1985.
#Nomenclature*]
[@ Key characters: Spores formed singly or in very loose, small clusters.
Etymology: from the Latin: simi(laris) = similar; glomus = cluster.
#Description*]"""

        result = parse_yeda_string(text)

        # Count by label
        labels = [r[0] for r in result]
        self.assertEqual(labels.count('Misc-exposition'), 2)
        self.assertEqual(labels.count('Nomenclature'), 3)
        self.assertEqual(labels.count('Description'), 2)

    def test_parse_no_yeda_blocks(self):
        """Test parsing text with no YEDDA blocks."""
        text = "Just plain text without any YEDDA annotations"
        result = parse_yeda_string(text)

        self.assertEqual(len(result), 0)

    def test_parse_malformed_block(self):
        """Test parsing with malformed YEDDA blocks."""
        # Missing closing bracket
        text = "[@ Some text\n#Label*"
        result = parse_yeda_string(text)
        self.assertEqual(len(result), 0)

        # Missing label
        text = "[@ Some text\n*]"
        result = parse_yeda_string(text)
        self.assertEqual(len(result), 0)


class TestYedaParserIntegration(unittest.TestCase):
    """Integration tests that require PySpark."""

    @classmethod
    def setUpClass(cls):
        """Set up Spark session for integration tests."""
        try:
            from pyspark.sql import SparkSession
            cls.spark = SparkSession.builder \
                .appName("YEDDA Parser Test") \
                .master("local[*]") \
                .config("spark.driver.host", "localhost") \
                .getOrCreate()
            cls.has_spark = True
        except ImportError:
            cls.has_spark = False

    @classmethod
    def tearDownClass(cls):
        """Stop Spark session."""
        if cls.has_spark:
            cls.spark.stop()

    def test_yeda_to_spark_df(self):
        """Test conversion to Spark DataFrame."""
        if not self.has_spark:
            self.skipTest("PySpark not available")

        from yeda_parser import yeda_to_spark_df

        text = "[@ Line 1\nLine 2\n#Label*]"
        df = yeda_to_spark_df(text, self.spark)

        self.assertEqual(df.count(), 2)
        self.assertEqual(len(df.columns), 3)
        self.assertIn('label', df.columns)
        self.assertIn('line', df.columns)
        self.assertIn('line_number', df.columns)

        # Check first row
        first_row = df.first()
        self.assertEqual(first_row['label'], 'Label')
        self.assertEqual(first_row['line'], 'Line 1')
        self.assertEqual(first_row['line_number'], 0)

    def test_yeda_to_spark_df_with_metadata(self):
        """Test conversion with metadata."""
        if not self.has_spark:
            self.skipTest("PySpark not available")

        from yeda_parser import yeda_to_spark_df
        import json

        text = "[@ Line 1\nLine 2\n#Label*]"
        metadata = {
            'source': 'test.txt',
            'version': '1.0',
            'count': 42
        }

        df = yeda_to_spark_df(text, self.spark, metadata)

        self.assertEqual(df.count(), 2)
        self.assertEqual(len(df.columns), 4)
        self.assertIn('metadata', df.columns)

        # Check metadata content
        first_row = df.first()
        parsed_meta = json.loads(first_row['metadata'])
        self.assertEqual(parsed_meta['source'], 'test.txt')
        self.assertEqual(parsed_meta['version'], '1.0')
        self.assertEqual(parsed_meta['count'], 42)

    def test_extract_metadata_field(self):
        """Test extracting specific field from metadata."""
        if not self.has_spark:
            self.skipTest("PySpark not available")

        from yeda_parser import yeda_to_spark_df, extract_metadata_field

        text = "[@ Line 1\n#Label*]"
        metadata = {'source_file': 'article.txt', 'version': '2.0'}

        df = yeda_to_spark_df(text, self.spark, metadata)
        df_extracted = extract_metadata_field(df, 'source_file')

        self.assertIn('source_file', df_extracted.columns)
        first_row = df_extracted.first()
        self.assertEqual(first_row['source_file'], 'article.txt')

    def test_yeda_file_auto_add_filepath(self):
        """Test auto-adding filepath to metadata."""
        if not self.has_spark:
            self.skipTest("PySpark not available")

        from yeda_parser import yeda_file_to_spark_df, extract_metadata_field

        # Using the test_data file
        df = yeda_file_to_spark_df(
            'test_data/article_reference.txt',
            self.spark,
            metadata={'version': '1.0'}
        )

        self.assertIn('metadata', df.columns)

        # Extract and check source_file
        df_extracted = extract_metadata_field(df, 'source_file')
        first_row = df_extracted.first()
        self.assertIn('article_reference.txt', first_row['source_file'])

        # Check custom metadata is also present
        df_extracted = extract_metadata_field(df, 'version')
        first_row = df_extracted.first()
        self.assertEqual(first_row['version'], '1.0')

    def test_get_label_statistics(self):
        """Test label statistics function."""
        if not self.has_spark:
            self.skipTest("PySpark not available")

        from yeda_parser import yeda_to_spark_df, get_label_statistics

        text = """[@ Line 1\nLine 2\n#Label1*]
[@ Line 3\nLine 4\nLine 5\n#Label2*]"""

        df = yeda_to_spark_df(text, self.spark)
        stats = get_label_statistics(df)

        self.assertEqual(stats.count(), 2)

        # Convert to list for easier testing
        stats_list = stats.collect()
        labels = {row['label']: row['count'] for row in stats_list}

        self.assertEqual(labels['Label1'], 2)
        self.assertEqual(labels['Label2'], 3)


if __name__ == '__main__':
    unittest.main()

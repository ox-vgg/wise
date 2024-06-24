import unittest
import json

from .query_parser import parse_search_query

class TestQueryParser(unittest.TestCase):
    def setUp(self):
        pass

    def test_query_parser(self):
        test_data = {
            'cooking food in video': {
                "cmd": "cooking food in video",
                "query": [
                    {
                        "query_str": [
                            "cooking food"
                        ],
                        "query_type": [
                            "text"
                        ],
                        "query_embedding_vec_op": [],
                        "search_target": "video",
                        "search_target_link": "in"
                    }
                ],
                "query_result_merge_ops": []
            },
            'car not in metadata':{
                "cmd": "car not in metadata",
                "query": [
                    {
                        "query_str": [
                            "car"
                        ],
                        "query_type": [
                            "text"
                        ],
                        "query_embedding_vec_op": [],
                        "search_target": "metadata",
                        "search_target_link": "not in"
                    }
                ],
                "query_result_merge_ops": []
            },
            '"cooking" in VIDEO AND "music" in AUDIO':{
                "cmd": "\"cooking\" in VIDEO AND \"music\" in AUDIO",
                "query": [
                    {
                        "query_str": [
                            "\"cooking\""
                        ],
                        "query_type": [
                            "text"
                        ],
                        "query_embedding_vec_op": [],
                        "search_target": "video",
                        "search_target_link": "in"
                    },
                    {
                        "query_str": [
                            "\"music\""
                        ],
                        "query_type": [
                            "text"
                        ],
                        "query_embedding_vec_op": [],
                        "search_target": "audio",
                        "search_target_link": "in"
                    }
                ],
                "query_result_merge_ops": [
                    "and"
                ]
            },
            'cooking IN VIDEO AND "background music" IN AUDIO or singing NOT IN metadata':{
                "cmd": "cooking IN VIDEO AND \"background music\" IN AUDIO or singing NOT IN metadata",
                "query": [
                    {
                        "query_str": [
                            "cooking"
                        ],
                        "query_type": [
                            "text"
                        ],
                        "query_embedding_vec_op": [],
                        "search_target": "video",
                        "search_target_link": "in"
                    },
                    {
                        "query_str": [
                            "\"background music\""
                        ],
                        "query_type": [
                            "text"
                        ],
                        "query_embedding_vec_op": [],
                        "search_target": "audio",
                        "search_target_link": "in"
                    },
                    {
                        "query_str": [
                            "singing"
                        ],
                        "query_type": [
                            "text"
                        ],
                        "query_embedding_vec_op": [],
                        "search_target": "metadata",
                        "search_target_link": "not in"
                    }
                ],
                "query_result_merge_ops": [
                    "and",
                    "or"
                ]
            },
            '@dog.jpg + "in snow" IN VIDEO':{
                "cmd": "@dog.jpg + \"in snow\" IN VIDEO",
                "query": [
                    {
                        "query_str": [
                            "dog.jpg",
                            "\"in snow\""
                        ],
                        "query_type": [
                            "file",
                            "text"
                        ],
                        "query_embedding_vec_op": [
                            "+"
                        ],
                        "search_target": "video",
                        "search_target_link": "in"
                    }
                ],
                "query_result_merge_ops": []
            },
            'animal - @cat.jpg IN Video and "wildlife safari" in MetaData':{
                "cmd": "animal - @cat.jpg IN Video and \"wildlife safari\" in MetaData",
                "query": [
                    {
                        "query_str": [
                            "animal",
                            "cat.jpg"
                        ],
                        "query_type": [
                            "text",
                            "file"
                        ],
                        "query_embedding_vec_op": [
                            "-"
                        ],
                        "search_target": "video",
                        "search_target_link": "in"
                    },
                    {
                        "query_str": [
                            "\"wildlife safari\""
                        ],
                        "query_type": [
                            "text"
                        ],
                        "query_embedding_vec_op": [],
                        "search_target": "metadata",
                        "search_target_link": "in"
                    }
                ],
                "query_result_merge_ops": [
                    "and"
                ]
            }
        }
        for query in test_data:
            parser_status, parsed_queries = parse_search_query(query)
            self.assertEqual(parser_status['status'], 'OK')
            self.assertEqual(parsed_queries, test_data[query])

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

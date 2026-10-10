import unittest

from autoresttest.utils import EmbeddingModel


class WordCaseTests(unittest.TestCase):
    def test_names_split_into_lowercase_words(self):
        cases = {
            "ownerId": "owner id",
            "Name": "name",
            "ID": "id",
            "userID": "user id",
            "HTTPServer": "http server",
            "IDs": "ids",
            "userIDs": "user ids",
            "URLsByID": "urls by id",
            "getHTTPResponseCode": "get http response code",
            "v2Api": "v api",
            "addressDTO": "address dto",
            "annotationJSON": "annotation json",
            "user_name": "user name",
            "pet-type": "pet type",
            "user.name": "user name",
            "filter[name]": "filter name",
            "__private__": "private",
            "address2": "address",
            "hg19Coordinates": "hg coordinates",
            "códigoPostal": "código postal",
            "co\u0301digoPostal": "código postal",  # decomposed accent
            "": "",
            "  ": "",
            "42": "",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(EmbeddingModel.handle_word_cases(name), expected)


if __name__ == "__main__":
    unittest.main()

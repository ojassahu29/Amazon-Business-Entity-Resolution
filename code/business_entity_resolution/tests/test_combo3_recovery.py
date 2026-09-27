import unittest

from evaluate_targeted_retrieval import combo3_candidate_variants


class Combo3RecoveryTests(unittest.TestCase):
    def test_method_b_preserves_baseline_and_adds_relaxed_match_evidence(self) -> None:
        country = "india"
        info_name = {"xylophone", "acme"}
        xylophone_postings = {f"noise-{index}" for index in range(499)}
        xylophone_postings.update({"name-recovery", "name-single", "name-multi"})
        alpha_postings = {f"address-noise-{index}" for index in range(500)}
        alpha_postings.update({"address-two", "address-geo", "number-info"})
        indexes = {
            "name_norm": {(country, "query name"): {"exact-name"}},
            "name_sorted": {},
            "name_compact": {},
            "prefix5": {(country, "query"): {"prefix-match"}},
            "name_tokens": {
                (country, "xylophone"): xylophone_postings,
                (country, "acme"): {"name-multi"},
            },
            "name_stopwords": {(country, "services"): {"name-recovery", "name-single"}},
            "addr_tokens": {
                (country, "alpha"): alpha_postings,
                (country, "bravo"): {"address-two"},
            },
            "geo_tokens": {
                (country, "mumbai"): {"number-geo", "address-geo"},
            },
            "addr_numbers": {
                (country, "1234"): {"number-geo", "number-info"},
            },
        }
        parsed = {
            "country": country,
            "name_norm": "query name",
            "name_sorted": "",
            "name_compact": "",
            "prefix5": "query",
            "info_name_tokens": info_name,
            "stop_name_tokens": {"services"},
            "info_addr_tokens": {"alpha", "bravo"},
            "geo_addr_tokens": {"mumbai"},
            "addr_numbers": {"1234"},
        }

        variants = combo3_candidate_variants(parsed, indexes)

        self.assertEqual(
            variants["method_b"],
            {"exact-name", "name-multi", "number-info", "prefix-match"},
        )
        self.assertEqual(
            variants["method_b_plus_name_relaxation"],
            variants["method_b"] | {"name-recovery", "name-single"},
        )
        self.assertEqual(
            variants["method_b_plus_address_relaxation"],
            variants["method_b"] | {"address-two", "number-geo"},
        )
        self.assertEqual(
            variants["method_b_plus_geographic_relaxation"],
            variants["method_b"] | {"number-geo", "address-geo"},
        )
        self.assertEqual(
            variants["method_b_plus_all_relaxations"],
            variants["method_b"]
            | {"name-recovery", "name-single", "address-two", "number-geo", "address-geo"},
        )


if __name__ == "__main__":
    unittest.main()

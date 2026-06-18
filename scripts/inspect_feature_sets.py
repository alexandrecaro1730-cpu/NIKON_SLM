"""
Inspect scenario feature sets before final submission.

Business objective
------------------
Confirm that each model scenario uses only features available at the correct
production decision time.

Coding objective
----------------
Print selected features for pre-build, in-build, and post-build scenarios.
"""

from lpbf_quality.features.feature_sets import get_feature_set
from lpbf_quality.config.settings import ID_COLUMNS, POST_BUILD_FEATURES, TARGET_COLUMN


def main() -> None:
    for scenario in ["prebuild", "inbuild", "postbuild"]:
        features = get_feature_set(scenario)

        print()
        print("=" * 80)
        print(f"Scenario: {scenario}")
        print(f"Number of features: {len(features)}")
        print("-" * 80)

        for feature in features:
            print(feature)

        forbidden = set(ID_COLUMNS + [TARGET_COLUMN])
        forbidden_found = sorted(forbidden.intersection(features))

        postbuild_found = sorted(set(POST_BUILD_FEATURES).intersection(features))

        print("-" * 80)
        print(f"Forbidden ID/target found: {forbidden_found}")

        if scenario in ["prebuild", "inbuild"]:
            print(f"Post-build features found: {postbuild_found}")
            assert not postbuild_found, (
                f"{scenario} should not include post-build features."
            )

        assert not forbidden_found, (
            f"{scenario} should not include ID or target columns."
        )

    print()
    print("Feature-set inspection passed.")


if __name__ == "__main__":
    main()
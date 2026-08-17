"""Generate a realistic synthetic university library assessment dataset."""

from pathlib import Path
import csv
import random


SEED = 42
N_RESPONDENTS = 300


LEVELS = [
    "Undergraduate",
    "Graduate",
    "Faculty/Staff",
]


AREAS = [
    "Engineering",
    "Computer Science",
    "Humanities",
    "Business",
    "Life Sciences",
    "Social Sciences",
]


FREQUENCIES = [
    "Daily",
    "Several times a week",
    "Weekly",
    "Monthly",
    "Rarely",
]


# -------------------------------------------------------------------
# Feedback pools
#
# The language intentionally overlaps across categories. This prevents
# the classifier from simply memorizing a single keyword such as
# "database" or "study room".
# -------------------------------------------------------------------

FEEDBACK_POOLS = {

    "Research Support": [
        "I can usually find useful academic material, although I sometimes need more guidance on where to begin.",
        "The resources are extensive, but I would benefit from more help choosing the right sources for a project.",
        "I appreciate the assistance available, though finding research advice at the right time can be difficult.",
        "The library offers good support for research, but more workshops on advanced research methods would be useful.",
        "I was able to complete my research, although I spent quite a while figuring out which resources were appropriate.",
        "Research consultations have been helpful when I have used them.",
        "I would like clearer guidance on evaluating sources and deciding which materials are reliable.",
        "The staff are knowledgeable, but I do not always know when or how to ask for research assistance.",
        "I found the materials I needed after some searching, but a short guide for new researchers would help.",
        "More examples of how to conduct research using the library's resources would be valuable.",
        "The research help I received was excellent and made the assignment much easier.",
        "I sometimes feel overwhelmed by the number of academic resources available.",
        "The library could provide more support for students who are unfamiliar with scholarly research.",
        "The workshops are useful, although they are not always offered when I need them.",
        "I had a positive experience getting help with a difficult research question.",
    ],

    "Digital Resources": [
        "Most of the material I need is available online, but finding the correct resource can take several attempts.",
        "Accessing electronic material from off campus is generally convenient, although the process is not always clear.",
        "The online collection is useful, but the search experience could be easier for someone using it for the first time.",
        "I can usually access articles without a problem, but some links seem to lead me in circles.",
        "The digital resources are extensive and generally work well for my coursework.",
        "Sometimes I find a useful article but have difficulty understanding which access option I should select.",
        "The online services save me a lot of time, although the website could explain its search options better.",
        "I would like a simpler way to locate electronic books and journal articles related to a course.",
        "The search tools are helpful once I understand them, but they are not very intuitive initially.",
        "Electronic materials have been convenient, especially when I cannot visit the library in person.",
        "I occasionally run into access issues when using library materials remotely.",
        "The online catalog has a lot of information, but it can be difficult to tell which result is most useful.",
        "I like being able to access academic resources from home.",
        "The digital collection is strong, although some materials take too many steps to access.",
        "I was able to find most of what I needed online without asking for assistance.",
    ],

    "Study Space": [
        "The library is a comfortable place to work, although finding a quiet seat can be difficult at busy times.",
        "I like the study areas, but there are not always enough places available during exams.",
        "The environment is generally good for studying, especially when the building is not crowded.",
        "More spaces for small groups would make the library more useful for collaborative projects.",
        "I appreciate the quiet areas, although some locations become noisy during the afternoon.",
        "The study environment works well for me, but seating availability varies considerably by time of day.",
        "The library has comfortable spaces, but I sometimes have to look around for an available desk.",
        "I would like more outlets near the desks where students usually work.",
        "The extended hours are helpful when I have assignments to finish late at night.",
        "The building provides a good environment for focused work, although group spaces can be difficult to reserve.",
        "During busy periods, the available seating does not always meet demand.",
        "I like studying here because there are several different types of work areas.",
        "The quiet areas are useful, but the distinction between quiet and collaborative spaces could be clearer.",
        "The physical environment is pleasant, though some seats are less comfortable for long sessions.",
        "I usually find somewhere to work, but peak hours can be frustrating.",
    ],

    "Collections": [
        "I usually find relevant material, although some older or specialized items are difficult to locate.",
        "A few of the books I needed for coursework were not available in the format I wanted.",
        "The collection is broad, but there are occasional gaps in materials for specialized subjects.",
        "I sometimes have to look outside the university for older academic sources.",
        "The available materials have supported most of my coursework successfully.",
        "I would appreciate more electronic versions of books that are frequently assigned in courses.",
        "Some required readings were difficult to obtain because only limited copies were available.",
        "The library has a strong selection overall, although certain subjects have fewer resources than others.",
        "I found most of the material I needed, but a few important sources were unavailable.",
        "More current editions of some textbooks would be helpful.",
        "The collection has been useful for my assignments and research.",
        "I occasionally discover that a recommended book is not accessible when I need it.",
        "There are plenty of resources, but it can be difficult to know whether the library owns a particular older item.",
        "I would like to see more materials added for interdisciplinary research.",
        "The selection is good, although availability depends heavily on the subject.",
    ],
}


def generate_dataset():

    random.seed(SEED)

    rows = []

    for i in range(1, N_RESPONDENTS + 1):

        level = random.choices(
            LEVELS,
            weights=[55, 35, 10],
        )[0]

        academic_area = random.choice(AREAS)

        frequency = random.choices(
            FREQUENCIES,
            weights=[8, 20, 28, 28, 16],
        )[0]

        base_score = {
            "Daily": 4.4,
            "Several times a week": 4.1,
            "Weekly": 3.8,
            "Monthly": 3.5,
            "Rarely": 3.0,
        }[frequency]

        satisfaction = max(
            1,
            min(
                5,
                round(
                    random.gauss(
                        base_score,
                        0.65,
                    ),
                    1,
                ),
            ),
        )

        digital_resources = max(
            1,
            min(
                5,
                round(
                    random.gauss(
                        base_score,
                        0.70,
                    ),
                    1,
                ),
            ),
        )

        study_space = max(
            1,
            min(
                5,
                round(
                    random.gauss(
                        base_score,
                        0.75,
                    ),
                    1,
                ),
            ),
        )

        research_support = max(
            1,
            min(
                5,
                round(
                    random.gauss(
                        base_score,
                        0.65,
                    ),
                    1,
                ),
            ),
        )

        # Select a theme and a less predictable feedback statement.
        theme = random.choice(
            list(FEEDBACK_POOLS.keys())
        )

        feedback = random.choice(
            FEEDBACK_POOLS[theme]
        )

        rows.append(
            [
                f"R{i:04d}",
                level,
                academic_area,
                frequency,
                satisfaction,
                digital_resources,
                study_space,
                research_support,
                feedback,
                theme,
            ]
        )

    return rows


def main():

    root = Path(__file__).resolve().parents[1]

    data_directory = root / "data"

    data_directory.mkdir(
        exist_ok=True
    )

    output_file = (
        data_directory
        / "library_survey.csv"
    )

    rows = generate_dataset()

    headers = [
        "respondent_id",
        "student_level",
        "academic_area",
        "library_usage_frequency",
        "satisfaction_score",
        "digital_resources_score",
        "study_space_score",
        "research_support_score",
        "open_feedback",
        "validation_theme",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)

        writer.writerows(rows)

    print(
        f"Generated {len(rows)} survey responses."
    )

    print(
        f"Saved dataset to: {output_file}"
    )


if __name__ == "__main__":
    main()
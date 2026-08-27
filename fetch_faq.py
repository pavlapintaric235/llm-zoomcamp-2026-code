import requests


def main() -> None:
    # This URL contains a list of all available courses.
    docs_url = "https://datatalks.club/faq/json/courses.json"

    response = requests.get(docs_url, timeout=30)
    response.raise_for_status()

    courses_raw = response.json()

    print(f"Number of courses: {len(courses_raw)}")

    # We will put all FAQ documents into this list.
    documents = []

    url_prefix = "https://datatalks.club/faq"

    # Download the FAQ documents for every course.
    for course in courses_raw:
        course_url = f"{url_prefix}{course['path']}"

        print(f"Downloading: {course_url}")

        course_response = requests.get(course_url, timeout=30)
        course_response.raise_for_status()

        course_data = course_response.json()

        # Add the course FAQ documents to our main list.
        documents.extend(course_data)

    print(f"\nTotal number of FAQ documents: {len(documents)}")

    # Display the first FAQ document.
    print("\nFirst document:")
    print(documents[0])


if __name__ == "__main__":
    main()
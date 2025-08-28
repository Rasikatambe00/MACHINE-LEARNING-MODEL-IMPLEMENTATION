# MACHINE-LEARNING-MODEL-IMPLEMENTATION
*COMPANY*: CODTECH IT SOLUTIONS
*NAME*: Rasika Nagesh Tambe
*INTERN ID*: CT06DZ73
*DOMAINA*: PYTHON
*DURATION*: 6 WEEK
*MENTOR* : NEELA SANTOSH



***

# Spam Detection Using Naive Bayes

This project implements a simple **spam detection system** for SMS messages using Python's scikit-learn library and a small labeled dataset. It uses **Natural Language Processing (NLP)** techniques and a Multinomial Naive Bayes classifier.

## Features

- Classifies SMS messages as **Spam** or **Ham (not spam)**.
- Uses **CountVectorizer** to convert messages into numerical features.
- Provides accuracy and detailed classification report.
- Includes custom message prediction examples.

## Dataset

The dataset (`spam.csv`) consists of short SMS messages labeled as spam or ham. Sample messages:
- "Win a FREE iPhone now! Click the link" — spam
- "Hey are we meeting today?" — ham

## How It Works

1. Loads and preprocesses the SMS dataset.
2. Splits data into training and test sets.
3. Transforms messages into bag-of-words features.
4. Trains a Multinomial Naive Bayes model.
5. Evaluates model accuracy and F1-score.
6. Tests predictions on custom messages.

## Usage

### Requirements

- Python 3.x
- pandas
- scikit-learn

### Files

- `spam_detection.py` / `task-4-code.txt`: Core code to train and test the spam detection model.
- `spam.csv`: SMS dataset with labels.

### Running the Code

1. Clone the repo and install dependencies:
   ```bash
   pip install pandas scikit-learn
   ```
2. Place `spam.csv` in the project directory.
3. Run the script:
   ```bash
   python spam_detection.py
   ```

### Output Example

- Accuracy and classification report are printed.
- Predictions for custom messages:
  - "Congratulations! You won a free prize!" → SPAM
  - "Hi, are you free tomorrow for a meeting?" → HAM
  - "Urgent! Claim your gift card now!" → SPAM

## License

This project is open-source and available under the MIT License.

***

Copy and use this README.md in your GitHub repo as needed.[2][3][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86273464/c93d980c-8e1c-42b1-ad3e-f2a1539d9b4a/task-4-small-data-set.txt)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86273464/c3b6f9fe-4dcf-47e9-a712-100f799143bc/task-4-code.txt)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86273464/479d7fac-6018-45aa-aeee-52b386929276/spam_detection.py)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86273464/4b0804bf-d097-4299-a59a-55736fd7cb9d/spam.csv)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/86273464/b0faa5c6-19dd-40a9-935f-2859224ff392/Screenshot-2025-08-28-105415.jpg?AWSAccessKeyId=ASIA2F3EMEYEVGOL2Y3H&Signature=5hrRTQKl9nr7adKMRY5FMxH6eJU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEEoaCXVzLWVhc3QtMSJGMEQCICBMqd5v17okbJYyM0uruA6f9%2FYYuGEO9jK83F3CjYAIAiAy9TW8XHi%2BkInInqS1qTeRKOvmFimq1oruDBYgMx%2BE%2FCr6BAii%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMCjK8eVRqjq4kXYaRKs4EDJTh%2FNNUOfbft97V4XihZq%2BdWr3%2BCIddbmWiN9P2w1aT3hlCBKKpcuRk3oj0NRdGE0FYoBOa1CgkP8ctwjnmck76xXezEPwrRvy1WmLEvqpF%2FT%2BZdntU3jDMWTr3cx7lVkKPwOSW38tNRX%2BAcbpE%2Fqcs9OJF2oywfMhkJMCKU1uYbFIIV3kPN567k2ce12fDAiwg8MNYjF7t9XEwUNjosp1uUuhYMlCNoZR58YQW8Pz%2Fjhe3oGwBsSqPXrGEYaSROjlf1kD1sjzlOCeT5%2F6hcpNVik1efOvGF1RaZRvtM6NzENDSOGRVzy9Afgl6ooTb0oF31RdxpWJuiyYxN1jgEu3t0WHuCC6mHNkNn7F8QzkQkx2yuA8y%2BADph%2BEURUd05yn%2Bil0VdadYjpilCbMTzt2PoSKB5LiXNHBXYCf7uXzIX689HY84b3AuQjf2zOI3wDjdh4FiXdJPoypVENSImDrU7oNorWOxVmMTTNkHBN6oqMpWw9jnYKRjUsZmhaapMIh80Y8uxwE9p%2Bo%2BnS6HfnhfCNUrR%2FSMOx%2Bv2HS823GpO2tH%2B91XCa4e4J6ChgxhQ1UL3zsS40cdOWQmN1LjVgI6MwNfpc0%2FYIjHWuSHjPjg6hHxyMqfEt1EU7jDOgHe46o1J42rA4luTHTMGfsp1DtnroUhlh%2BMXdMzuQlGlt%2B48DM9Vn6UAXABeKTKlu16%2F1jL3o9ENo4tBz2vUrLIAu6P86Y5bOoKyfqDHYzNB3dpMyuAuD5QKJ%2BB%2ByHgor58t1%2FGw2uqbkYL77Nv4Qgwv73AxQY6mwFbgObsQEuGDR67gSD6BVNTk%2BAG1xqVHH9%2BzlBhw3TwC0U3eY%2Foq7Uj4SzH4JhAyMdmhEVvc6c%2Fve7PGox5%2Bw0TbStb80rFUamaHk2ewkwFw4rxkel7wNL20x%2B8PX%2ByxU1vDvENY1XePLuBVm6RJjVKrmgX0%2FO%2FhmJWTYS8Q%2F%2FQX49g12JGixaQRtOh9FmQgflCgc%2FNKOI1NxCNOg%3D%3D&Expires=1756375340)

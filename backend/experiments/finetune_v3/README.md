# Bilingual LLM Counseling Framework

This repository contains the system implementation and supporting experimental resources for the research article:

**A Bilingual RAG-Grounded Counseling Framework with Feedback-Driven Contextual Bandit Adaptation for University Students**

Our framework integrates a locally deployed large language model (LLM), retrieval-augmented generation (RAG), strategy-conditioned fine-tuning, and feedback-driven contextual bandit adaptation to provide bilingual (English and Korean) counseling support for university students.

## Naming Clarification

**Nova** is the internal development name of the counseling system. In the accompanying MDPI manuscript, the system is referred to as **"our system"** or **"our framework"** rather than Nova.

Internal filenames, model identifiers, and directories retain the original Nova naming convention.

## Repository Structure

The project is organized into two main components:

- **Frontend:** Contains the user interface of the counseling application.
- **Backend:** Contains the core system implementation, including LLM integration, retrieval-augmented generation (RAG), counseling logic, and feedback-driven adaptation.

The backend also contains two important research directories:

- **`backend/finetune_v3/`** — Contains the strategy-conditioned fine-tuning implementation, dataset preparation scripts, and model training resources.
- **`backend/Experiment/`** — Contains the experimental evaluation scripts, model comparison configurations, and supporting evaluation resources used in the study.

## Code and Data Availability

This repository accompanies the MDPI manuscript and contains the counseling system implementation, fine-tuning scripts, and supporting experimental resources.

The general counseling datasets used in this study—**CounselChat, Amod Mental Health Counseling Conversations, ESConv, MentalChat16K, Dolly-15k, and Career Guidance QA**—are publicly available from their respective original sources.

Additional implementation files, experimental resources, and documentation may be incorporated as the repository is finalized.
## Associated Publication

*A Bilingual RAG-Grounded Counseling Framework with Feedback-Driven Contextual Bandit Adaptation for University Students.*

Manuscript prepared for submission to an MDPI journal.

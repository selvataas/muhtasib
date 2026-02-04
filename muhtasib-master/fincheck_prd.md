Write a comprehensive Product Requirements Document (PRD) for implementing the "FinCheck" pipeline from the Compliance-to-Code framework, adapted for Turkish financial regulations, as described in the research paper at https://arxiv.org/abs/2505.19804. Structure the PRD in markdown format with the following sections: 

1. **Overview**: A high-level summary of the system, its purpose (automating compliance checking for Turkish regulations like those from CMB and BRSA), and how it adapts the original Chinese-focused framework to Turkish (handling Turkish language, local data sources like KAP at kap.org.tr, and multilingual LLMs).

2. **Goals and Objectives**: List key goals, such as improving RegTech efficiency, ensuring traceability, and reducing manual compliance efforts. Include success metrics like Pass@1 for code generation accuracy.

3. **Target Users**: Describe users (e.g., compliance officers, legal teams in Turkish financial firms) and use cases (e.g., checking share repurchase compliance against CMB rules).

4. **Features and Modules**: Detail the four core modules (Structure Predictor, Code Generator, Information Retriever, Report Generator) with sub-sections for each, including:
   - Description of functionality.
   - Input/output specs.
   - Turkish-specific adaptations (e.g., handling Turkish text, local APIs, legal terms like "hisse geri alımı").
   - Integration with the full pipeline.

5. **Technical Requirements**: 
   - Programming language: Python 3.10+.
   - Key dependencies: transformers (Hugging Face), pandas, requests.
   - LLM assumptions: Use placeholders for fine-tuned multilingual models (e.g., Turkish-adapted Qwen or DeepSeek-Coder; note that actual fine-tuning is out of scope, but include loading via pipeline).
   - Data handling: Use pandas DataFrames for company data.
   - Security: Emphasize safe code execution and human oversight for legal use.
   - Deployment: Suggest as a script or Streamlit web app.

6. **Implementation Details**: For each module, provide complete, functional Python code in code blocks. Base it on this module breakdown:
   - **Structure Predictor**: Parses raw Turkish regulation text into Compliance Units (CUs: subject, condition, constraint, contextual_info) and relations using a fine-tuned LLM with Turkish prompts.
   - **Code Generator**: Generates executable Python code from CUs, with chain-of-thought prompting for compliance logic.
   - **Information Retriever**: Fetches Turkish company data (e.g., via mock KAP API or synthetic data) into a DataFrame.
   - **Report Generator**: Executes the code on data and generates a Turkish-language report using an LLM.
   Include a main function to chain all modules into a full pipeline. Use synthetic examples for testing (e.g., a sample regulation text and company data).

7. **Testing and Evaluation**: Outline unit tests, error handling, and evaluation metrics (e.g., F1 for structure extraction, CodeBLEU for code similarity).

8. **Risks and Limitations**: Discuss legal risks (e.g., not a substitute for expert advice), data privacy, and needs for ongoing updates to Turkish regulations.

9. **Next Steps**: Suggest fine-tuning LLMs on a Turkish dataset, deployment options, and iterations.

Ensure the PRD is practical, with code that is executable (handle errors gracefully), and focused on modularity. Keep the tone professional and include disclaimers for real-world use.

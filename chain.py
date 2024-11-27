import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import PyPDF2 as pdf

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile",
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context is too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def input_pdf_text(uploaded_file):
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in range(len(reader.pages)):
            page = reader.pages[page]
            text += str(page.extract_text())
        return text

    def write_mail(self, job, links, resume_text):
        prompt_email = PromptTemplate.from_template(
            """
            Job Description: {job_description}

            Resume:{resume_text}

            Project Links: {link_list}

            Instructions:

            You are an AI assistant tasked with crafting a personalized cold email to a potential client based on their job description and the candidate's qualifications.

            Analyze the job description and identify key requirements, skills, and experiences.

            Compare the candidate's resume to the job description, highlighting relevant skills, experiences, and projects.

            Tailor the cold email to emphasize the candidate's strengths and how they align with the specific needs of the job.

            Use the provided project links to support your claims and demonstrate the candidate's capabilities.

            Use the candidate's name throughout the email, wherever appropriate, instead of using a placeholder 

            Ensure the email is concise, professional, and persuasive.
            
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke(
            {
                "job_description": str(job),
                "link_list": links,
                "resume_text": resume_text,
            }
        )
        return res.content

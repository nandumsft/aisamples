# aisamples

https://www2.gov.bc.ca/gov/content/environment/air-land-water/water

https://www2.gov.bc.ca/gov/content/environment/air-land-water/water/water-licensing-rights/water-allocation-notations

https://www2.gov.bc.ca/assets/gov/environment/air-land-water/water/water-licensing-and-rights/imapbc_how_to_identify_water_allocation_notations.pdf
https://www2.gov.bc.ca/assets/gov/environment/air-land-water/water/water-licensing-and-rights/joint_works_agreement_water_licence.pdf

from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01"
)
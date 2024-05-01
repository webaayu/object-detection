import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

def authenticate():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            "service_account_key.json",
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        drive_service = build('drive', 'v3', credentials=credentials)
        return drive_service
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return None

def list_files_in_folder(folder_id):
    drive_service = authenticate()
    if drive_service:
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(name)",
            pageSize=10
        ).execute()
        items = results.get('files', [])
        return [item['name'] for item in items]
    else:
        return []

def main():
    st.title("Google Drive Folder File List")
    folder_id = st.text_input("Enter Google Drive folder ID:")
    submit = st.button("List Files")

    if submit:
        if folder_id.strip():
            st.write("Fetching files...")
            files = list_files_in_folder(folder_id)
            if files:
                st.write("Files in the folder:")
                for file in files:
                    st.write(file)
            else:
                st.error("Failed to fetch files. Please check the folder ID.")
        else:
            st.warning("Please enter a valid folder ID.")

if __name__ == "__main__":
    main()


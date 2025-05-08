# File Upload Configuration

This document describes the configuration for file uploads in the application, including size limits and best practices.

## File Size Limits

The application is configured to handle large file uploads, with the following limits:

- **Frontend (React Dropzone)**: 5GB maximum file size
- **Backend (FastAPI)**: 5GB maximum request body size

These limits can be adjusted if needed by modifying the following files:

### Frontend

- `src/components/ui/drop-zone.tsx`: Default maxSize parameter (5GB)
- `src/components/ui/upload-file-modal.tsx`: Explicit maxSize parameter (5GB)

### Backend

- `middleware/file_upload.py`: LargeFileUploadMiddleware max_upload_size parameter (5GB)
- `server.py`: FastAPI middleware configuration

## Implementation Details

### Frontend

The frontend uses react-dropzone for file uploads, which has a configurable maximum file size. We've set this to 5GB by default.

```tsx
// Default configuration in drop-zone.tsx
maxSize = 5 * 1024 * 1024 * 1024, // 5GB max size
```

### Backend

The backend uses a custom middleware to increase the FastAPI maximum request body size limit, which is 100MB by default.

```python
# Configuration in server.py
add_large_file_upload_middleware(app, max_upload_size=5 * 1024 * 1024 * 1024)  # 5GB
```

## Best Practices for Large File Uploads

When uploading large files (>1GB), consider the following best practices:

1. **Network Stability**: Ensure a stable network connection when uploading large files.
2. **Progress Indication**: The UI provides feedback on upload progress.
3. **Timeout Settings**: Server timeout settings have been increased to accommodate large uploads.
4. **Concurrency Limits**: The server limits concurrent connections to prevent overload during large uploads.

## Troubleshooting

If you encounter issues with large file uploads:

1. **"File is too large" error**: This indicates that the file exceeds the configured size limit (5GB).
2. **Timeout errors**: For very large files, the server might time out. Consider breaking the file into smaller chunks.
3. **Network errors**: Check your network connection and try again.

## Future Improvements

For even larger files (>5GB), consider implementing:

1. **Chunked uploads**: Breaking files into smaller chunks and reassembling them on the server.
2. **Resumable uploads**: Allowing uploads to be paused and resumed.
3. **Direct-to-storage uploads**: Bypassing the application server and uploading directly to the storage backend.

import { Box, Typography, Link } from "@mui/material";
import { Description, PictureAsPdf } from "@mui/icons-material";

// Import all resource files using Vite's glob
const resourceFiles = import.meta.glob('/src/resources/**/*.{html,pdf,tex,md,apkg}', { 
  eager: true,
  as: 'url'
});

const Resources = () => {
  const courseFiles = {};
  
  Object.entries(resourceFiles).forEach(([path, url]) => {
    const match = path.match(/\/src\/resources\/([^\/]+)\/(.+)$/);
    if (!match) return;
    
    const [, course, filePath] = match;
    const filename = filePath.split('/').pop();
    const extension = filename.split('.').pop()?.toLowerCase();
    
    if (!['html', 'pdf', 'tex', 'md', 'apkg'].includes(extension)) return;
    
    if (!courseFiles[course]) {
      courseFiles[course] = [];
    }
    
    courseFiles[course].push({
      name: filename,
      type: extension,
      url: url
    });
  });

  // Sort files within each course
  Object.keys(courseFiles).forEach(course => {
    courseFiles[course].sort((a, b) => a.name.localeCompare(b.name));
  });

  // Sort courses
  const sortedCourses = Object.keys(courseFiles).sort();

  const getFileIcon = (fileType) => {
    if (fileType === 'pdf') {
      return <PictureAsPdf sx={{ fontSize: 16, mr: 1 }} />;
    }
    return <Description sx={{ fontSize: 16, mr: 1 }} />;
  };

  return (
    <Box sx={{ 
      width: "100%", 
      height: "100%", 
      padding: 4, 
      overflowY: "auto" 
    }}>
      <Typography variant="h6" sx={{ mb: 4, textTransform: 'capitalize' }}>
        Resources
      </Typography>
      <Typography sx={{ mb: 4 }}>
      A more <i>comprehensive</i> repository of materials can be found <a href="https://github.com/haezera/haezera.github.io/tree/main/src/resources" target="_blank" rel="noopener noreferrer" style={{ textDecoration: "underline", color: "#4A9EFF" }}>here</a>
      </Typography>
      <Box sx={{ width: "100%", height: "50px", border: "1px solid #ff3045", backgroundColor: "rgb(255, 48, 69, 0.4)",  borderRadius: 2, p: 2, display: "flex", alignItems: "center", justifyContent: "center", mb: 3 }}>
        I graduated in December of 2025! The below notes thus could be outdated, or just plain wrong.
      </Box>
      
      {sortedCourses.map(course => {
        const files = courseFiles[course];
        if (!files || files.length === 0) return null;

        return (
          <Box key={course} sx={{ mb: 4 }}>
            <Typography 
              variant="h6" 
              sx={{ 
                mb: 2, 
                textTransform: 'uppercase',
                fontWeight: 600
              }}
            >
              {course}
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {files.map((file, index) => (
                <Link
                  key={`${course}-${index}`}
                  href={file.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    color: '#4A9EFF',
                    textDecoration: 'underline',
                    backgroundColor: 'rgba(74, 158, 255, 0.1)',
                    padding: '8px 12px',
                    borderRadius: '4px',
                    '&:hover': {
                      backgroundColor: 'rgba(74, 158, 255, 0.2)',
                    }
                  }}
                >
                  {getFileIcon(file.type)}
                  <Typography variant="body2">
                    {file.name}
                  </Typography>
                </Link>
              ))}
            </Box>
          </Box>
        );
      })}
    </Box>
  );
};

export default Resources;

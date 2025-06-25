import os
import re
from bs4 import BeautifulSoup

def extract_plotly_dimensions(plotly_file_path):
    """Extract width and height from Plotly HTML file."""
    try:
        with open(plotly_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Enhanced patterns for Plotly dimension detection
        width_patterns = [
            r'\"width\":\s*(\d+)',
            r'\'width\':\s*(\d+)',
            r'width:\s*(\d+)',
            r'layout.*?\"width\".*?(\d+)',
            r'config.*?toImageButtonOptions.*?width.*?(\d+)',
            r'responsive.*?width.*?(\d+)',
            r'autosize.*?width.*?(\d+)'
        ]
        
        height_patterns = [
            r'\"height\":\s*(\d+)',
            r'\'height\':\s*(\d+)', 
            r'height:\s*(\d+)',
            r'layout.*?\"height\".*?(\d+)'
        ]
        
        width = None
        height = None
        
        # Try to find width with more aggressive searching
        for pattern in width_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                # Take the largest reasonable width found
                widths = [int(w) for w in matches if 300 <= int(w) <= 2000]
                if widths:
                    width = max(widths)
                    break
        
        # Try to find height
        for pattern in height_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                heights = [int(h) for h in matches if 200 <= int(h) <= 1500]
                if heights:
                    height = max(heights)
                    break
        
        # Alternative: look for div with plotly plot dimensions
        if not width or not height:
            div_pattern = r'<div[^>]*id=["\']([^"\']*plot[^"\']*)["\'][^>]*style=["\'][^"\']*width:\s*(\d+)px[^"\']*height:\s*(\d+)px'
            div_match = re.search(div_pattern, content, re.IGNORECASE)
            if div_match:
                width = width or int(div_match.group(2))
                height = height or int(div_match.group(3))
        
        # Fallback: reasonable defaults for financial charts
        final_width = width or 1000
        final_height = height or 600
        
        print(f"Extracted dimensions: {final_width}x{final_height} from {plotly_file_path}")
        return final_width, final_height
        
    except Exception as e:
        print(f"Error reading Plotly file {plotly_file_path}: {e}")
        return 1000, 600

def replace_static_images_with_plotly_iframes(html_file_path, output_path=None, image_scale=0.8, buffer_width=50, increase_font = 3):
    """
    Replace static images in an HTML report with interactive plotly iframes.
    Automatically adjusts document width based on actual plot dimensions.
    
    This function scans an HTML report for image tags, and if there is a matching
    HTML file with the same name (excluding extension) in the same directory,
    it replaces the static image with an interactive plotly iframe.
    
    **Args:**
        - html_file_path (str): Path to the input HTML report file
        - output_path (str, optional): Path for the output HTML file with interactive plots. 
                If None, will use the original filename with '_interactive' suffix.
        - image_scale (float, optional): Scale factor for static images (0.1 to 1.0). Default is 0.8 (80%).
        - buffer_width (int, optional): Additional buffer width to add to the document for better layout. Default is 50px.
        - increase_font (int, optional): Increase font size for better readability in landscape mode. Default is 3px. 
            This is added to the base font size of 11px.
    
    Returns:
        str: Path to the new HTML file with interactive plots
    """

    import os
    from bs4 import BeautifulSoup
    
    # Get directory of the input file
    input_dir = os.path.dirname(html_file_path)
    if input_dir == "":
        input_dir = "."
    
    # Look for interactive figures in the figures subdirectory
    figures_dir = os.path.join(input_dir, "figures")
    
    # Set default output path if not provided
    if output_path is None:
        base_name = os.path.basename(html_file_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(input_dir, f"{name}_interactive{ext}")
    
    # Get list of HTML files in the figures directory
    html_files = []
    html_file_names = []
    
    if os.path.exists(figures_dir):
        html_files = [f for f in os.listdir(figures_dir) if f.endswith('.html')]
        html_file_names = [os.path.splitext(f)[0] for f in html_files]
        print(f"Found {len(html_files)} HTML files in figures/ directory that could be interactive plots")
    else:
        print(f"Warning: figures/ directory not found at {figures_dir}")
        figures_dir = input_dir  # Fallback to input directory if figures dir doesn't exist
    
    # Read the input HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Track maximum plot width for document sizing
    max_plot_width = 1150  # Default minimum
    plot_dimensions = []
    
    # Find all image tags and collect plot dimensions
    img_tags = soup.find_all('img')
    print(f"Found {len(img_tags)} image tags in HTML report")
    
    # Keep track of replacements
    replacements = 0
    
    # Validate image_scale parameter
    image_scale = max(0.1, min(1.0, image_scale))
    image_scale_percent = int(image_scale * 100)
    
    # Process each image tag
    for img in img_tags:
        if 'src' in img.attrs:
            # Get image source path
            src = img['src']
            # Get image filename without extension
            img_filename = os.path.splitext(os.path.basename(src))[0]
            
            # Check if there's a matching HTML file
            if img_filename in html_file_names:
                plotly_file_path = os.path.join(figures_dir, f"{img_filename}.html")
                width, height = extract_plotly_dimensions(plotly_file_path)
                plot_dimensions.append((width, height))
                max_plot_width = max(max_plot_width, width)
                
                print(f"Found matching interactive plot for image: {img_filename} ({width}x{height})")
                
                # Create new iframe tag with proper sizing using style attribute
                iframe = soup.new_tag('iframe')
                iframe['src'] = f"./figures/{img_filename}.html"
                iframe['class'] = 'plotly-iframe'
                iframe['frameborder'] = '0'
                iframe['scrolling'] = 'no'
                # Use style attribute instead of width/height attributes to ensure proper sizing
                iframe['style'] = f'width: {width}px; height: {height}px; max-width: 100%; border: none; display: block; margin: 20px auto;'
                
                # Replace image tag with iframe
                img.replace_with(iframe)
                replacements += 1
    
    print(f"Replaced {replacements} static images with interactive plots")
    print(f"Maximum plot width detected: {max_plot_width}px")
    
    # Update or create dynamic CSS based on actual plot dimensions
    head = soup.find('head')
    if head:
        # Remove existing plotly-iframe style if present
        existing_styles = head.find_all('style')
        for style_tag in existing_styles:
            if style_tag.string and '.plotly-iframe' in style_tag.string:
                style_tag.decompose()
        
        # Calculate proper document width with margins
        text_width = max_plot_width + 200  # Text wider than plots
        min_content_width = max(1000, max_plot_width + 100)

        # Simplified and more robust CSS
        dynamic_css = f"""
        /* Plotly iframe styling */
        .plotly-iframe {{
            border: none;
            margin: 20px auto;
            display: block;
            max-width: 100%;
            background: white;
            overflow: hidden;
        }}
        
        iframe.plotly-iframe {{
            min-width: 800px;
            min-height: 500px;
        }}

        /* Enhanced font sizes for landscape viewing */
        p {{ font-size: {11 + increase_font}px !important; }}
        li {{ font-size: {11 + increase_font}px !important; }}
        figcaption {{ font-size: {11 + increase_font}px !important; }}
        table {{ font-size: {10 + increase_font}px !important; }}
        math, .math {{ font-size: {11 + increase_font}px !important; }}
        code, pre {{ font-size: {11 + increase_font}px !important; }}
        
        h1 {{ font-size: {42 + increase_font}px !important; }}
        h2 {{ font-size: {33 + increase_font}px !important; }}
        h3 {{ font-size: {26 + increase_font}px !important; }}
        h4 {{ font-size: {20.5 + increase_font}px !important; }}
        h5 {{ font-size: {17.5 + increase_font}px !important; }}
        h6 {{ font-size: {15 + increase_font}px !important; }}
        
        /* Center-aligned tables and images */
        table {{
            width: auto;
            min-width: 50%;
            max-width: {max_plot_width}px;
            margin: 20px auto;
            text-align: center;
        }}
        
        table th, table td {{
            text-align: center !important;
        }}
        
        img {{
            width: {image_scale_percent}% !important;
            height: auto !important;
            max-width: {max_plot_width}px;
            display: block !important;
            margin: 20px auto !important;
            object-fit: contain;
        }}
        
        figure {{
            text-align: center;
            margin: 20px auto;
        }}
        
        /* Dynamic content width - will be adjusted by JavaScript */
        .content-container {{
            max-width: {text_width}px;
            margin: 0 auto;
            padding: 0 20px;
            box-sizing: border-box;
        }}
        
        /* Override default markdown preview styles */
        html body[for=html-export]:not([data-presentation-mode]) .markdown-preview {{
            max-width: none !important;
            width: 100% !important;
            padding: 2em 20px !important;
            margin: 0 !important;
        }}
        
        /* Wrap content in container */
        .markdown-preview > * {{
            max-width: {text_width}px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        /* Ensure plots can be wider than text */
        .markdown-preview > iframe.plotly-iframe,
        .markdown-preview > figure,
        .markdown-preview > table {{
            max-width: {max_plot_width + 100}px !important;
        }}
        
        /* TOC adjustments */
        html body[for=html-export]:not([data-presentation-mode])[html-show-sidebar-toc] .md-sidebar-toc {{
            width: 280px !important;
        }}
        
        html body[for=html-export]:not([data-presentation-mode])[html-show-sidebar-toc] .markdown-preview {{
            left: 280px !important;
            width: calc(100% - 280px) !important;
            padding: 2em 20px !important;
        }}
        
        /* Minimum body width to prevent cramping */
        body {{
            min-width: {min_content_width}px;
        }}
        """
        
        # Add JavaScript for dynamic width adjustment
        js_script = f"""
        <script>
        (function() {{
            const MAX_PLOT_WIDTH = {max_plot_width};
            const TEXT_WIDTH = {text_width};
            const MIN_CONTENT_WIDTH = {min_content_width};
            
            function adjustContentWidth() {{
                const viewportWidth = window.innerWidth;
                const tocVisible = document.body.hasAttribute('html-show-sidebar-toc');
                const tocWidth = tocVisible ? 280 : 0;
                const availableWidth = viewportWidth - tocWidth;
                
                // Calculate optimal content width
                let contentWidth;
                if (availableWidth > TEXT_WIDTH + 200) {{
                    contentWidth = TEXT_WIDTH;
                }} else if (availableWidth > MIN_CONTENT_WIDTH + 100) {{
                    contentWidth = Math.min(TEXT_WIDTH, availableWidth - 100);
                }} else {{
                    contentWidth = Math.max(MIN_CONTENT_WIDTH, availableWidth - 40);
                }}
                
                // Apply the width
                const style = document.getElementById('dynamic-width-style') || document.createElement('style');
                style.id = 'dynamic-width-style';
                style.textContent = `
                    .markdown-preview > * {{
                        max-width: ${{contentWidth}}px !important;
                    }}
                    .markdown-preview > iframe.plotly-iframe,
                    .markdown-preview > figure,
                    .markdown-preview > table {{
                        max-width: ${{Math.max(contentWidth, MAX_PLOT_WIDTH)}}px !important;
                    }}
                `;
                
                if (!document.getElementById('dynamic-width-style')) {{
                    document.head.appendChild(style);
                }}
            }}
            
            // Adjust on load and resize
            window.addEventListener('load', adjustContentWidth);
            window.addEventListener('resize', adjustContentWidth);
            
            // Watch for TOC toggle
            const tocBtn = document.getElementById('sidebar-toc-btn');
            if (tocBtn) {{
                tocBtn.addEventListener('click', function() {{
                    setTimeout(adjustContentWidth, 10);
                }});
            }}
            
            // MutationObserver to catch TOC attribute changes
            const observer = new MutationObserver(function(mutations) {{
                mutations.forEach(function(mutation) {{
                    if (mutation.type === 'attributes' && 
                        mutation.attributeName === 'html-show-sidebar-toc') {{
                        setTimeout(adjustContentWidth, 10);
                    }}
                }});
            }});
            
            observer.observe(document.body, {{
                attributes: true,
                attributeFilter: ['html-show-sidebar-toc']
            }});
            
            // Initial adjustment
            setTimeout(adjustContentWidth, 100);
        }})();
        </script>
        """
        
        plot_style = soup.new_tag('style')
        plot_style.string = dynamic_css
        head.append(plot_style)
        
        # Add JavaScript
        js_tag = soup.new_tag('script')
        js_tag.string = js_script.replace('<script>', '').replace('</script>', '')
        head.append(js_tag)
    
    # Write modified HTML to output file
    print(f"Saving interactive HTML report to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    print(f"Interactive HTML report saved to {output_path}")
    print(f"Document width optimized for plots up to {max_plot_width}px wide")
    print(f"Text width set to {text_width}px with dynamic adjustment")
    print(f"Static images scaled to {image_scale_percent}% width and center-aligned")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replace static images in an HTML report with interactive Plotly iframes.")
    parser.add_argument('html_file', help="Path to the input HTML report file")
    parser.add_argument('--output', help="Path for the output HTML file with interactive plots", default=None)
    parser.add_argument('--image-scale', type=float, default=0.8, help="Scale factor for static images (0.1 to 1.0). Default is 0.8 (80%%)")
    parser.add_argument('--buffer', type=float, default=80, help="Add additional buffer width to document (default 80px)")
    parser.add_argument('--increase-font', type=int, default=3, help="Increase font size for better readability in landscape mode (default 3px)")
    
    args = parser.parse_args()
    
    replace_static_images_with_plotly_iframes(args.html_file, args.output, args.image_scale, args.buffer, args.increase_font)

    #Sample usage:
    # python intplots_html_landscape.py '/Users/jamesbishop/Documents/Financial/Investment/MACRO_STUDIES/TwitterThreadz/hOdLeRs/42_Macro_BackTests.html' --image-scale 0.8 --increase-font 4
    #python intplots_html_landscape.py  "/Users/jamesbishop/Documents/Financial/Investment/MACRO_STUDIES/Proper_Studies/42Macro/BTC_Signals_Backtests/42_Macro_BackTests.html" --image-scale 0.8 --increase-font 3
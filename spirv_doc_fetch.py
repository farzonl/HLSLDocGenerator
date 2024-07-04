import requests
import pypandoc
import os
import sys
import re
import copy
from bs4 import BeautifulSoup

spirv_vulkan3_base_url = 'https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/'

def replace_rel_links_with_hyperlinks(markdown_content, url):
    # Define a regex pattern to match Markdown links
    pattern = r'\[([^\]]+)\]\(#([^\)]+)\)'


    # Replace Markdown links with hyperlinks
    def replace_link(match):
        title = match.group(1)
        id = match.group(2) 
        return f'[{title}]({url}#{id})'
    
    # Perform the replacement
    replaced_content = re.sub(pattern, replace_link, markdown_content)
    return replaced_content

def replace_href_with_hyperlinks(markdown_content, url):
    # Define a regex pattern to match Markdown links
    pattern = r'\[([^\]]+)\]\(([^\)]+)\)'


    # Replace Markdown links with hyperlinks
    def replace_link(match):
        title = match.group(1)
        id = match.group(2) 
        return f'[{title}]({url}{id})'
    
    # Perform the replacement
    replaced_content = re.sub(pattern, replace_link, markdown_content)
    return replaced_content

def fetch_html(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.text

def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup

def get_table_id(table):
    rows = table.find_all('tr')
    if not rows:
        return (False, '')
        
    first_row = rows[0]
    first_cell = first_row.find('td')
    if not first_cell:
        return (False, '')
    
    table_id = first_cell.find('a')
    if not table_id or not table_id.get('id'):
        return (False, '')
    
    table_id = table_id['id']
    return (True, table_id)

def delete_first_line(text):
    newline_index = text.find('\n')
    if newline_index != -1:
        modified_text = text[newline_index + 1:]
        return modified_text
    return text

def extract_glsl_tables(soup, url):
    tables = soup.find_all('tbody')
    id_to_markdown = {}

    table_header = BeautifulSoup("""
        <tr>
        <th>Number</th>
        <th>Operand 1</th> 
        <th>Operand 2</th>  
        <th>Operand 3</th>
        <th>Operand 4</th>
        </tr>
        """, 'html.parser').tr

    for table in tables:
        trs = table.find_all('tr')
        table_id = trs[0].find('strong').text.strip()
        if table_id == 'Extended Instruction Name':
            continue
        if not table_id:
            continue
    
        # Check if there are at least 2 <tr> tags
        if len(trs) > 1:
            table_tag = soup.new_tag('table')
            table_tag.append(table_header)
            tr_copy = copy.copy(trs[1])
            table_tag.append(tr_copy)
            trs[1].decompose()
            trs[0].insert_after(table_tag)
        
         # Convert HTML to Markdown
        markdown = convert_html_to_markdown(table)
        markdown = f"# {table_id}:\n\n## Description:\n{markdown}\n"
        markdown =  replace_rel_links_with_hyperlinks(markdown, url)

        id_to_markdown[table_id] = markdown
    
    return id_to_markdown


def check_for_dual_ids(id: str):
    # Regular expression pattern
    pattern = r'(\S+NV) \((\S+KHR)\)'
    
    match = re.search(pattern, id)
    if match:
        return (True, match.group(1), match.group(2))
    return (False, '', '')

NV_to_KHR_map = { 'OpReportIntersectionNV' : 'OpReportIntersectionKHR',
                  'OpTypeAccelerationStructureNV' : 'OpTypeAccelerationStructureKHR'
                }

def extract_tables(soup, url):
    tables = soup.find_all('tbody')
    id_to_markdown = {}
    
    table_header = BeautifulSoup("""
        <tr>
        <th>Word Count</th>
        <th>Opcode</th> 
        <th>Results</th>  
        <th>Operands</th>
        </tr>
        """, 'html.parser').tr
    for table in tables:
        valid_table, table_id = get_table_id(table)
        if not valid_table:
            continue
        
         # Find all <tr> tags within the <tbody>
        trs = table.find_all('tr')
    
        # Check if there are at least 2 <tr> tags
        if len(trs) > 1:
            table_tag = soup.new_tag('table')
            table_tag.append(table_header)
            tr_copy = copy.copy(trs[1])
            table_tag.append(tr_copy)
            trs[1].decompose()
            trs[0].insert_after(table_tag)

        # Convert HTML to Markdown
        markdown = convert_html_to_markdown(table)
        markdown = delete_first_line(markdown)
        markdown = f"# [{table_id}]({url}#{table_id}):\n\n## Description:\n{markdown}\n"
        markdown =  replace_rel_links_with_hyperlinks(markdown, url)

        #has_ids, id1, id2 = check_for_dual_ids(table_id)
        #if has_ids:
        #    id_to_markdown[id1] = markdown
        #    id_to_markdown[id2] = markdown
        #    print(f"ids id 1: {id1}  id 2: {id2}")
        #else:
        #    id_to_markdown[table_id] = markdown
        id_to_markdown[table_id] = markdown
        
        if table_id in NV_to_KHR_map:
            id_to_markdown[NV_to_KHR_map[table_id]] = markdown
        
    
    return id_to_markdown

def download_pandoc():
    """Download pandoc if not already installed"""
    try:
        # Check whether it is already installed
        pypandoc.get_pandoc_version()
    except OSError:
        # Pandoc not installed. Let's download it silently.
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            pypandoc.download_pandoc()
            sys.stdout = sys.__stdout__
    
            # Hack to delete the downloaded file from the folder,
            # otherwise it could get accidently committed to the repo
            # by other scripts in the repo.
            pf = sys.platform
            if pf.startswith('linux'):
                pf = 'linux'
            url = pypandoc.pandoc_download._get_pandoc_urls()[0][pf]
            filename = url.split('/')[-1]
            os.remove(filename)

def convert_html_to_markdown(html_content):
    download_pandoc()

    try:
        # Convert HTML to Markdown using pypandoc
        markdown_content = pypandoc.convert_text(html_content, 'markdown', format='html', extra_args=['--to=gfm'])
        return markdown_content
    except Exception as e:
        print(f"Error converting HTML to Markdown: {e}")
        return None

def parse_spirv_uni_spec():
    url = 'https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html'
    html = fetch_html(url)
    soup = parse_html(html)
    id_to_markdown = extract_tables(soup, url)
    return id_to_markdown

def parse_spirv_glsl_spec():
    url = 'https://registry.khronos.org/SPIR-V/specs/1.0/GLSL.std.450.html'
    html = fetch_html(url)
    soup = parse_html(html)
    id_to_markdown = extract_glsl_tables(soup, url)
    return id_to_markdown

def parse_spirv_vulkan3_spec():
    id_to_markdown = {}
    html = fetch_html(spirv_vulkan3_base_url)
    soup = parse_html(html)
    for tr in soup.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) > 1:
            a_tag = tds[1].find('a')
            if a_tag:
                key = a_tag.text.strip().replace('.html', '')
                url = spirv_vulkan3_base_url + a_tag['href']
                id_to_markdown[key] = url
    return id_to_markdown

def parse_spirv_vulkan_man_page(url):
    html = fetch_html(url)
    markdown = convert_html_to_markdown(html)
    markdown =  replace_href_with_hyperlinks(markdown, spirv_vulkan3_base_url)
    return markdown


def parse_spirv_spec():
    spirv_doc = parse_spirv_uni_spec()
    spirv_glsl_doc = parse_spirv_glsl_spec()
    spirv_doc.update(spirv_glsl_doc)
    return spirv_doc

def main():
    #id_to_markdown = parse_spirv_vulkan3_spec()
    #for table_id, markdown in id_to_markdown.items():
    #    print(markdown)
    print(parse_spirv_vulkan_man_page('https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkSetEvent.html'))

if __name__ == "__main__":
    main()

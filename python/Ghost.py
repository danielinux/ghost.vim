from langchain_core.tools import tool
import json
import requests
import tree_sitter_c
import os
import ollama
import re
from rich import print

last_query_time = 0


class Tool:
    def get_tools(self):
        """Return all methods decorated with @tool in this class."""
        return [getattr(self.__class__, attr) for attr in dir(self.__class__)
                if callable(getattr(self.__class__, attr)) and hasattr(getattr(self.__class__, attr), 'name')]

    def fix_path(self, path):
        if path == '.':
            return path
        while path.startswith('/') or path.startswith('.'):
            path = path[1:]
        if not path.startswith(self.base_dir):
            path = f"{self.base_dir}/{path}"
        while path.endswith('/') or path.endswith('*') or path.endswith('.'):
            path = path[:-1]
        return path

    def handle_tool_call(self, call):
        """
        Generic handler for langchain tool calls. Must match name to method decorated with @tool.
        """
        name = call.function.name
        args = call.function.arguments
        for attr in dir(self):
            method = getattr(self, attr)
            if callable(method) and hasattr(method, 'name') and method.name == name:
                return method.run(args)
        return f"Error: tool method '{name}' not found"

# --- WebSearch Tool ---
class WebSearch(Tool):
    def __init__(self):
        super().__init__()

    @staticmethod
    @tool
    def web_search(query: str) -> str:
        """Search for a topic using DuckDuckGo. Returns a summary of top matching results."""
        """Search the web for the given query. This tool will return the best matching results on the web. This is only a preview: I will retrieve the information by reading the web pages."""
        from duckduckgo_search import DDGS
        # print(f"** WebSearch executing search for: {query}")
        results = DDGS().text(query, max_results=5)
        ctx = [
               {
                   "url" : x['href'],
                   "title": x['title'],
                   "preview": x['body']
               } for x in results
            ]
        return json.dumps(ctx)

    @staticmethod
    @tool
    def web_browse(url: str, pos: int = 0, size: int = 1024) -> str:
        """Download and extract visible text from a web page. Use position and size to page through long content."""
        """Retrieve the content of a web page from a given URL. Use pos and len to scroll through the HTML code in large pages, and focus on your research."""
        import urllib.request
        from bs4 import BeautifulSoup
        if size > 2048:
            size = 2048
        try:
            #print(f"** WebSearch fetching URL: {url}")
            content = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            return str(text[pos:pos+size])
        except Exception as e:
            return f"Error fetching URL: {e}"

# --- Container ---
class Container:
    def __init__(self, name):
        self.name = name

    def run(self, argv):
        import subprocess
        return subprocess.run(['podman', 'exec', self.name] + argv, capture_output=True, text=True)

# --- Workspace Tool ---
class Workspace(Tool):
    def __init__(self, container):
        super().__init__()
        self.container = container

    @tool
    def run(self, argv: list[str]) -> str:
        """Run a shell command in the workspace machine as the 'developer' user. After the commands executes, if an error is returned I think about what is wrong. Am I in the right path? Am I missing information I can retrieve with a different call?"""
        try:
            result = self.container.run(argv)
            result = subprocess.run(argv, capture_output=True, text=True)
            return result.stdout + " " + result.stderr
        except Exception as e:
            return f"Error running command: {e}"

    @staticmethod
    @tool
    def read_file(path: str, pos: int = 0, size: int = 1024) -> str:
        """Read a section of a text file starting from a specific byte offset for a fixed number of bytes."""
        """Read file content starting from a specific position for a given size."""
        try:
            if not os.path.exists(path) or not os.path.isfile(path):
                return "Error: File does not exist or is not a file."
            with open(path, 'rb') as f:
                import re
                f.seek(pos)
                content = f.read(size)
            return content.decode(errors='replace')
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    @tool
    def write_file(path: str, content: str, pos: int = 0, mode: str = 'rewrite') -> str:
        """Write text to a file with control over position and mode (rewrite, insert, or append)."""
        """Write content to a file. Supports append, insert or replace modes."""
        try:
            with open(path, 'rb+' if os.path.exists(path) else 'wb+') as f:
                f.seek(pos)
                if mode == 'rewrite':
                    f.write(content.encode())
                    f.truncate(pos + len(content))
                elif mode == 'insert':
                    existing = f.read()
                    f.seek(pos)
                    f.write(content.encode() + existing)
                elif mode == 'append':
                    f.seek(0, os.SEEK_END)
                    f.write(content.encode())
                else:
                    return "Error: Invalid mode selected."
            return f"File '{path}' written successfully."
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    @tool
    def truncate_file(path: str, size: int) -> str:
        """Cut off a file's contents after the specified byte length."""
        """Truncate a file to a new size."""
        try:
            with open(path, 'r+') as f:
                f.truncate(size)
            return f"File '{path}' truncated to {size} bytes."
        except Exception as e:
            return f"Error truncating file: {e}"

    @staticmethod
    @tool
    def append_file(path: str, content: str) -> str:
        """Add new text to the end of a file without removing its current contents."""
        """Append content to the end of a file."""
        try:
            with open(path, 'a') as f:
                f.write(content)
            return f"Content appended to file '{path}'."
        except Exception as e:
            return f"Error appending file: {e}"

# --- DocReader Tool ---
from langchain_core.tools import tool
from collections import defaultdict

class DocReaderTool(Tool):
    from collections import defaultdict

    current = None
    available = []

    @tool
    def pdf_open(path: str) -> str:
        """Load a PDF into memory. This must be called before accessing document content."""
        """Once a PDF is open, it will stay open until a new document replaces it. """
        DocReaderTool.current = DocReaderTool(path)
        return f"[DocReader] Loaded PDF: {path}"

    @tool
    def pdf_lookup_freetext(string: str, method: str = "ANY") -> str:
        """Search for text across all pages of the current PDF. Returns the top 20 pages with the most matches. Method can be 'ANY' (default), which returns pages with any matches, or 'ALL', which only returns pages where all occurrences of the string are found."""
        if not DocReaderTool.current:
            return "No PDF document is currently open. Use pdf_open() first."

        page_hits = []
        toc = DocReaderTool.current.doc.toc
        toc_map = defaultdict(str)

        for level, title, page_num in toc:
            toc_map[page_num] = title

        for i, chunk in enumerate(DocReaderTool.current.doc.md_chunked):
            lines = chunk.page_content.splitlines()
            count = sum(1 for line in lines if string in line)
            if count > 0:
                page = i + 1
                page_hits.append((page, count))

        if method == "ALL":
            page_hits = [entry for entry in page_hits if entry[1] > 1]

        page_hits.sort(key=lambda x: -x[1])
        top_hits = page_hits[:20]

        ret = "".join(f"*** Page {p}: {hits} hits\n" for p, hits in top_hits)
        return ret

    @tool
    def get_page(index: int) -> str:
        """Return the content of a specific page of the currently loaded PDF."""
        if not DocReaderTool.current:
            return "No PDF document is currently open. Use pdf_open() first."
        pages = DocReaderTool.current.doc.md_chunked
        if index < 0 or index >= len(pages):
            return f"Page index {index} is out of range. Document has {len(pages)} pages."
        #print ("Content: " +pages[index].page_content)
        return pages[index].page_content

    def __init__(self, path: str):
        import pymupdf4llm
        import pymupdf
        from langchain.text_splitter import MarkdownHeaderTextSplitter

        class DocReaderDocument():
            def __init__(self, path):
                self.pdfdoc = pymupdf.Document(path)
                self.toc = []  # disabled TOC for whole-doc processing

                def toc_hdrinfo(toc, span, page: pymupdf.Page) -> str:
                    hdr = [e for e in toc if span['text'].strip().lower() in e[1].strip().lower()]
                    if len(hdr) == 0:
                        return ""
                    page_num = page.number + 1 if page.number else 0
                    hdr = [e for e in hdr if e[2] == page_num]
                    if len(hdr) == 0:
                        return ""
                    return '#'*hdr[0][0] + ' '

                from langchain.text_splitter import RecursiveCharacterTextSplitter

                text = pymupdf4llm.to_markdown(self.pdfdoc)
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                self.md_chunked = [type('Chunk', (object,), {'page_content': chunk}) for chunk in splitter.split_text(text)]

            def get_by_header(self, header: str) -> str:
                hlw = header.lower()
                chunks = [c for c in self.md_chunked if hlw in [v.lower() for v in c.metadata.values()]]
                chunks += [c for c in self.md_chunked if any([v.lower() for v in c.metadata.values() if v.lower() in hlw])]
                return "".join([c.page_content for c in chunks])

        self.doc = DocReaderDocument(path)




# --- LlamaScope Tool ---

from tree_sitter import Language, Parser
import shutil

class CodeLookup:
    def __init__(self, base_dir):
        self.files = []
        self.symbol_table = {}
        self.base_dir = os.path.expanduser(base_dir)
        self.base_dir = os.path.abspath(self.base_dir)
        self.ghost_dir = os.path.join(base_dir, ".ghost")
        try:
            os.mkdir(self.ghost_dir)
        except:
            pass
        self.parser = Parser(Language(tree_sitter_c.language()))
        self.ghost_reset()

    def refresh(self):
        self.files = []
        self.symbol_table = {}
        os.chdir(self.ghost_dir)
        self.process_directory('.')
        STRUCT_RE = re.compile(
                r'\b(?P<kind>struct|union|enum)\s+'         #kind
                r'(?:(?:\w+)\s+)*?'                         # optional macros like PACKED
                r'(?P<name>\w+)\s*'                         # name
                r'\{(?P<body>.*?)\}\s*;',                   # body inside {}, lazy match
                re.DOTALL
        )

        # Add types with decorators (tree-sitter fails to detect those)
        for fil in self.files:
            if (os.path.isdir(fil)):
                continue
            with open(fil, 'rb') as f:
                source_code = f.read().decode('utf-8', errors='replace')
            matches = STRUCT_RE.finditer(source_code)
            for match in matches:
                kind = match.group('kind')
                name = match.group('name')
                body = match.group('body')
                start = match.start()
                end = match.end()
                line = source_code[0:start].count('\n') + 1
                found = False
                for s in self.symbol_table:
                    if self.symbol_table[s]['name'] == name:
                        found = True
                        break
                if not found:
                    extended_uid = fil + ":" + name + ":" + str(line)
                    fn = {'name':name, 'type': kind, 'file':fil, 'line':line , 'start': start , 'end': end}
                    self.symbol_table[extended_uid] = fn
                    #print(f'Added symbol {kind} {name} from {fil} {str(line)} via re module')


    def ghost_reset(self):
        # Recursively copy the original sources in the ghost dir
        for root, dirs, files in os.walk(self.base_dir):
            if '.ghost' in dirs:
                dirs.remove('.ghost')
            if '.git' in dirs:
                dirs.remove('.git')
            for file in files:
                if file.endswith(".c") or file.endswith(".h") or file.endswith(".pdf"):
                    src_file_path = os.path.join(root, file)
                    dst_file_path = src_file_path.replace(self.base_dir, self.ghost_dir)
                    dst_dir = os.path.dirname(dst_file_path)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    shutil.copy2(src_file_path, dst_file_path)
        self.refresh()

    def ghost_apply(self, patch):
        file = patch['path']
        line = patch['line']
        removing = patch.get('removing', '')
        adding = patch.get('adding','')
        if (len(adding) > 0) and not adding.endswith('\n'):
            adding += '\n'
        if (len(removing) > 0) and not removing.endswith('\n'):
            removing += '\n'


        rem_lines = removing.count('\n')
        print('lines to remove: ', str(rem_lines))

        with open(file, 'rb+') as f:
            content = f.read().decode(errors='replace')
        lines = content.splitlines()
        if removing in content and len(patch['removing']) > 0:
            idx = content.index(removing)
            sz = len(removing)
            prefix = content[0:idx - 1]
            suffix = content[idx + sz:]
            context = adding
        else:
            prefix = '\n'.join(lines[:line])
            context = adding
            suffix = '\n'.join(lines[line:]) + '\n'

        # Update the symbols for the modified file
        file = file.replace(self.ghost_dir, '.')

        if file not in self.files:
            self.files.append(file)
        with open(file, 'wb') as f:
            content = prefix + '\n'  + context + suffix
            print(f'Prefix: {prefix[-40:]}')
            print(f'Context: {context}')
            print(f'Suffix: {suffix[:40]}')

            f.write(content.encode())
        while True:
            found = False
            for s in self.symbol_table:
                if self.symbol_table[s]['file'] == file:
                    self.symbol_table.pop(s)
                    found = True
                    break
            if not found:
                break
            if os.path.getsize(file) == 0:
                try:
                    os.remove(file)
                except:
                    pass
                self.files.remove(file)
        symbols = self.extract_symbols(content, file)
        self.symbol_table.update(symbols)

    def ghost_accept_all(self):
        # Recursively copy the modified sources in the base dir
        for root, dirs, files in os.walk(self.ghost_dir):
            if '.ghost' in dirs:
                dirs.remove('.ghost')
            if '.git' in dirs:
                dirs.remove('.git')
            for file in files:
                if file.endswith(".c") or file.endswith(".h"):
                    src_file_path = os.path.join(root, file)
                    dst_file_path = src_file_path.replace(self.ghost_dir, self.base_dir)
                    dst_dir = os.path.dirname(dst_file_path)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    shutil.copy2(src_file_path, dst_file_path)


    def get_parser(self):
        return self.parser
    def get_files(self):
        return self.files
    def get_symbol_table(self):
        return self.symbol_table
    # Function to read file
    def read_file(self, file_path, max_=0):
        content = None
        """Read the contents of a file."""
        with open(file_path, "rb") as f:
            content = f.read()
        if max_ > 0:
            content = content[:max_]
        return content.decode(errors='replace')

    def extract_symbols(self, source_code, file_path):
        """Extract function definitions using tree-sitter."""
        tree = self.parser.parse(source_code.encode("utf-8"))
        root_node = tree.root_node
        symbols = {}

        def find_symbol(node):
            # Handle typedefs"
            if node.type == 'type_definition':
                kind = 'typedef'
                name = node.text.decode('utf-8', errors='replace')
                return (kind, name, node)

            # Handle plain structs, enums, unions
            if node.type in ('struct_specifier', 'union_specifier', 'enum_specifier'):
                kind = node.type.replace('_specifier', '')  # struct/union/enum
                identifier = None
                has_fields = False
                for child in node.children:
                    if child.type in ('identifier', 'type_identifier'):
                        identifier = child.text.decode('utf-8', errors='replace')
                    elif child.type == 'field_declaration_list':
                        has_fields = True
                if not has_fields:
                    return ('', None, None)
                return (kind, identifier or '<anonymous>', node)

            # Handle normal typedef
            if node.type == 'type_definition':
                for child in node.children:
                    if child.type == 'type_identifier':
                        return ('typedef', child.text.decode('utf-8', errors='replace'), node)

            # Macros
            if node.type in ('preproc_def', 'preproc_function_def'):
                for child in node.children:
                    if child.type == 'identifier':
                        return ('macro', child.text.decode('utf-8', errors='replace'), node)

            # Functions
            for child in node.children:
                if child.type in ('function_declarator', 'declarator'):
                    for i in child.children:
                        if i.type == "identifier":
                            return ('function', i.text.decode("utf-8", errors='replace'), node)
                elif child.type == 'parenthesized_declarator':
                    return find_symbol(child)

            return ('', None, None)
        def visit(node):
            """Recursively visit nodes to find function definitions, abstract types, globals."""
            symbol_name = None
            if (node.type == "function_definition" or
                node.type == "declaration" or
                node.type == "struct_specifier" or
                node.type == "enum_specifier" or
                node.type == "union_specifier" or
                node.type == "type_definition" or
                node.type == "preproc_def" or
                node.type == "preproc_function_def"):
                symbol_type, symbol_name, def_node = find_symbol(node)
            else:
                pass

            if symbol_name:
                start_line = def_node.start_point[0] + 1  # Convert to 1-based indexing
                extended_uid = file_path + ":" + symbol_name + ":" + str(start_line)
                fn = { 'name':symbol_name, 'type': symbol_type, 'file':file_path, 'line':start_line , 'start': def_node.start_byte , 'end':def_node.end_byte}
                symbols[extended_uid] = fn

            for child in node.children:
                visit(child)

        visit(root_node)
        return symbols

    # Process directory recursively
    def process_directory(self, directory):
        """Recursively process a directory and parse C/H files."""

        for root, dirs, files in os.walk(directory):
            for d in dirs:
                if (os.path.basename(d).startswith('.')):
                    continue
                if d.startswith('.'):
                    continue
                d_path = os.path.join(root, d)
                if '.git' in d_path:
                    continue
                print("adding directory: " + d_path)
                self.files.append(d_path + '/')

            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".c") or file.endswith(".h"):
                    self.files.append(file_path)
                    try:
                        source_code = self.read_file(file_path)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}. File skipped.")
                        continue
                    #print("parsing: " + file_path)
                    symbols = self.extract_symbols(source_code, file_path)
                    self.symbol_table.update(symbols)

        print(directory + ": " + str(len(self.symbol_table)) + " symbols from " + str(len(self.files)) + " files.")
        #print(json.dumps(self.files))
        #print(json.dumps(self.symbol_table, indent=4))

class ExtendedLlamaScope(CodeLookup, Tool):
    """Extended LlamaScope tool that includes additional file system operations and enhanced code lookup capabilities."""
    symbol_table = None
    files = None
    parser = None

    @staticmethod
    @tool
    def list_files(path: str) -> str:
        """List all files and directories in the specified path. Use '.' to start at the root."""
        msg = f'Listing all files and directores in {path}:'
        if os.path.exists(path) and os.path.isdir(path):
            for f in os.listdir(path):
                if os.path.isdir(f):
                    msg += '<DIR>    '+ f + '\n'
                else:
                    msg += '<FILE>   '+ f + '\n'
            if msg == '':
                msg += 'This directory is empty.'
            return msg
        else:
            return "Error: Path does not exist or is not a directory."

    @staticmethod
    @tool
    def list_symbols(path: str, match: str = "") -> str:
        """Show all symbol definitions in the given file, at the path 'path'. Use 'match' to filter by name."""
        matches = []
        for s in ExtendedLlamaScope.symbol_table:
            if (len(match) == 0 or match in ExtendedLlamaScope.symbol_table[s]['name']) and path == ExtendedLlamaScope.symbol_table[s]['file']:
                matches.append(ExtendedLlamaScope.symbol_table[s])
        return ' '.join(f"{i}. {s['name']} @ {s['path']}" for i, s in enumerate(matches))

    @staticmethod
    @tool
    def lookup_symbol(symbol: str, index: int = 0) -> str:
        """Return the source code of a function or symbol from the codebase by name and optional index."""
        matches = []
        for s in ExtendedLlamaScope.symbol_table:
            if ExtendedLlamaScope.symbol_table[s]['name'] == symbol:
                matches.append(ExtendedLlamaScope.symbol_table[s])
        if not matches:
            return f"Symbol '{symbol}' not found."
        if index >= len(matches):
            return f"Index {index} out of range for symbol '{symbol}'."
        symbol_info = matches[index] if index >= 0 else matches[0]
        try:
            with open(symbol_info['path'], 'rb') as f:
                f.seek(symbol_info['start_byte'])
                content = f.read(symbol_info['end_byte'] - symbol_info['start_byte'])
                return content.decode(errors='replace')
        except Exception as e:
            return f"Error reading symbol content: {e}"

    @staticmethod
    @tool
    def lookup_freetext(string, method):
        """Perform a freetext search for a given string within the codebase. Use method ANY or ALL to match any of the terms or the entire string respectively."""
        found = defaultdict(int)

        search_terms = string.strip().lower().split()  # Preprocess search terms

        for fp in ExtendedLlamaScope.files:
            if fp.endswith('/'):
                continue
            try:
                with open(fp, 'rb') as f:
                    for ln, line in enumerate(f, start=1):
                        lower_line = line.decode('utf-8', errors='replace').lower()
                        if method == 'ALL':
                            if all(term in lower_line for term in search_terms):
                                found[fp] += 1
                        elif method == 'ANY':
                            if any(term in lower_line for term in search_terms):
                                found[fp] += 1
            except (OSError, UnicodeDecodeError) as e:
                print(f"Error reading {fp}: {e}")  # Handle potential file read errors gracefully

        sorted_found_desc = dict(sorted(found.items(), key=lambda item: item[1], reverse=True))
        if len(sorted_found_desc) == 0:
            return "I could not find anything matching the string '"+string+"'.\n"
        elif (len(sorted_found_desc) > 20):
            ret = "There are too many files matching the search for '"+string+"'. Only showing the first 20 hits.\n"
            for i,x in enumerate(sorted_found_desc):
                if i < 20:
                    ret += x + ' ('+str(sorted_found_desc[x])+' lines matching)\n'
                else:
                    break
            return ret

        ret = ''
        if len(sorted_found_desc) == 0:
            return "I could not find anything matching the string '"+string+"'.\n"
        else:
            ret = 'Lines matching the string "' + string + '":\n'
        for x in sorted_found_desc:
            ret += x + ' ('+str(sorted_found_desc[x])+' lines matching)\n'
        return ret


    @staticmethod
    @tool
    def lookup_freetext_in_file(path, string, method):
        """Perform a freetext search for a given string within the existing file, which is part of the codebase. Use method ANY or ALL to match any of the terms or the entire string respectively."""
        found = 0
        search_terms = string.strip().lower().split()  # Preprocess search terms
        try:
            with open(path, 'r', encoding='utf-8') as f:
                ctx = ''
                for ln, line in enumerate(f, start=1):
                    lower_line = line.lower()
                    if method == 'ALL':
                        if all(term in lower_line for term in search_terms):
                            ctx += f"Line {ln}: {line}"
                    elif method == 'ANY':
                        if any(term in lower_line for term in search_terms):
                            ctx += f"Line {ln}: {line}"
                if ctx.count('\n') == 0:
                    return 'Could not find any lines in ' + path + ' matching the string "' + string + "'.\n"
                else:
                    return "Lines in '"+path+"' matching '"+string+"': \n" + ctx
        except (OSError, UnicodeDecodeError) as e:
            return str(e)
        return ctx

    @staticmethod
    @tool
    def get_file_content(path: str, start_line: int = 0, num_lines: int = 10) -> str:
        """Return a code snippet from a file starting at a given line, including a number of lines."""
        try:
            with open(path, 'r') as file:
                lines = file.readlines()
            return ''.join(lines[start_line:start_line + num_lines])
        except Exception as e:
            return f"Error getting file content: {e}"

    def __init__(self, base_dir):
        CodeLookup.__init__(self,base_dir)
        Tool.__init__(self)
        ExtendedLlamaScope.symbol_table = CodeLookup.get_symbol_table(self)
        ExtendedLlamaScope.files = CodeLookup.get_files(self)
        ExtendedLlamaScope.parser = CodeLookup.get_parser(self)

    def get_symbol(self, symbol: str) -> dict:
        """Retrieve a symbol's information from the symbol table."""
        for s in self.symbol_table:
            if self.symbol_table[s]['name'] == symbol:
                return self.symbol_table[s]

    def find_all_instances(self, text) -> list[str]:
        """Find all instances of a given text across files."""
        matches = []
        for file_path in self.files:
            if not os.path.isdir(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read().decode(errors='replace')
                    if text in content:
                        matches.append(file_path)
        return matches

    def get_function_code(self, symbol: str) -> str:
        match = None
        for s in self.symbol_table:
            if self.symbol_table[s]['name'] == symbol:
                match = self.symbol_table[s]
                break
        if not match:
            return None
        with open(match['file'], 'r') as f:
            f.seek(match['start'])
            content = f.read(match['end'] - match['start'])
            return content

    def get_references(self, references:list[str]) -> str:
        references = [x for x in references if x]
        refs = []
        seen = set()
        references = [x for x in references if not (x in seen or seen.add(x))]
        for ref in references:
            for s in self.symbol_table:
                if self.symbol_table[s]['name'] == ref:
                    sym = self.symbol_table[s]
                    with open(sym['file'], 'r') as f:
                        f.seek(sym['start'])
                        content = f.read(sym['end'] - sym['start'])
                        if content not in refs:
                            refs.append(content)
                    break
        print("[magenta]Found " + str(len(refs)) + " referenced functions[/]")
        return '\n'.join(x for x in refs)


    def get_types(self, adts:list[str]) -> str:
        types = []
        seen = set()
        adts = [x for x in adts if not (x in seen or seen.add(x))]

        for adt in adts:
            key,name = adt.split(' ') if ' ' in adt else ('typedef', adt)
            for s in self.symbol_table:
                if self.symbol_table[s]['name'] == name:
                    sym = self.symbol_table[s]
                    with open(sym['file'], 'r') as f:
                        f.seek(sym['start'])
                        content = f.read(sym['end'] - sym['start'])
                        if content not in types:
                            types.append(content)
        print("[magenta]Found " + str(len(adts)) + " types[/] (tot len: " + str(len(' '.join(types))) + ')')
        return '\n'.join(types)

    def get_callers(self, symbol:str) -> list:
        callers = []
        code = ''
        for s in self.symbol_table:
            sym = self.symbol_table[s]
            code += self.get_function_code(sym['name']) + '\n'
            if code and symbol in code and sym['name'] != symbol and sym['name'] not in callers:
                callers.append(sym['name'])
            if len(callers) >= 5:
                break
        print("[magenta]Found " + str(len(callers)) + " callers[/]")
        #print(str(callers))
        return code

    def get_called_functions(self, symbol: str) -> list:
        code = self.get_function_code(symbol)
        if not code:
            return ''
        try:
            tree = self.parser.parse(code.encode())
            root = tree.root_node
            calls = []

            def walk(node):
                if node.type == "call_expression":
                    func_node = node.child_by_field_name("function")
                    if func_node:
                        name = code[func_node.start_byte:func_node.end_byte]
                        if name not in calls:
                            calls.append(name)
                for child in node.children:
                    walk(child)
            walk(root)

            print("[magenta]Found " + str(len(calls)) + " called functions[/]")
            if len(calls) > 5:
                calls = calls[0:5]
            print("[magenta](Limiting context to 5 called functions)[/]")
            code = ''
            for fn in calls:
                code += self.get_function_code(fn) + '\n'
            return code

        except Exception as e:
            print(f"[Tree-sitter error] {e}")
            return ''

    def byte_to_line(self, path: str, byte_offset: int) -> int:
        try:
            with open(path, 'rb') as f:
                content = f.read(byte_offset)
            return content.count(b'\n')
        except Exception:
            return 0

class PipeLine:
    def __init__(self, workspace = None, root_dir = '.'):
        self.workspace = workspace
        print('initializing llamascope\n')
        self.llamascope = ExtendedLlamaScope(root_dir)
        print('llamascope initialized\n')
        self.web = WebSearch()
        self.doc = DocReaderTool.__new__(DocReaderTool)  # lazy init; must call pdf_open first
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    DocReaderTool.available.append(file_path)
        print("[red]Available documents:[/] " + '\n'.join(DocReaderTool.available))


    def get_rules(self, category):
        rules_dir = os.path.join(os.path.dirname(__file__), 'rules')
        rules_path = os.path.join(rules_dir, category + '.md')
        try:
            with open(rules_path, 'r') as f:
                rules = f.read()
            return rules
        except:
            return ''

    def call_task_agent(self, prompt: str, model="qwen2.5-coder:32b"):
        json_path = os.path.join(os.path.dirname(__file__), 'task_list.json')
        with open(json_path, 'r') as f:
            task_list = f.read()

        rules = self.get_rules('design')
        if len(rules) > 0:
            rules = '\nHere are some mandatory rules to follow:\n' + rules + '\n'
        system_prompt = """
        You are a code transformation planner for a C codebase.
        Your job is to convert a short document containing of language instructions into a structured JSON list of tasks.
        To better define the tasks and decide their priorities you are given access to tools via function-call tools
        format, which you can use to explore the codebase.

        Each output task consists in a single action that can be performed in the codebase.
        Each output task affects only one single portion of the code in one file.

            <IMPORTANT> Your final answer MUST consist only of a valid JSON list of output tasks. Omit any comments, thoughts, explainations.</IMPORTANT>
        """ + rules + """


        The following is a list of available tasks:
        """ + task_list + """
        Each task MUST include:
        - \"type\": the type of transformation (see task list)
        - \"target\": the function, macro, type or symbol name
        - \"target_new\": The new name for the symbol or the file, if applicable
        - \"file\": the file where the transformation should be applied
        - \"details\": human-readable explanation of the task, giving details about the transformation requested
        - \"ADTs\": a list of Abstract Data Types (ADTs) that should be used in the transformation (typedefs, enums, structs, unions) and are ALREADY PRESENT in the codebase.
        - \"references\": a list of suggested symbols such as functions or macros names, that are useful to look at as reference, performing similar task or in the same domain, or using same tools types or algorithms. The references MUST BE symbols ALREADY PRESENT in the codebase. In case of new functions, types or macros, at least two references must be present.
        """
        prompt = """
        Work one step at a time, as follows:
        - Read and understand the input document.
        - Establish a solid knowledge of the code that will be affected by the changes. Check the types and functions involved in the changes that will be commanded.
        - Research: Look for any relevant symbols in the codebase, to reference or suggest in the output tasks fields. Only use symbols that you are sure are present in the codebase.
        - Loop: repeat the research step as many times as needed, until you are satisfied about the knowledge and you can proceed to generate the list of tasks.
        - If any of your tasks requires a new function to be implemented, ensure that a task to generate the symbol is included in the output BEFORE the task using the new function.
        - When all the fields can be populated rigorously and the tasks are finally ordered, output the list of tasks in the requested JSON format. Ensure that you only include tasks defined according to the task format.
        """ + prompt

        tools = self.llamascope.get_tools()
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.model_json_schema()
                }
            }
            for t in tools
        ]
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt + '\n' + 'OUTPUT JSON:\n'}
        ]
        #print(messages)
        while True:
            print("\n[green]Thinking...[/] 🤔💭 ", end = '\r')
            response = ollama.chat(
                model=model,
                messages=messages,
                tools=tool_defs,
                options={
                    "temperature": 0.1,
                    "top_p": 0.7,
                    "repeat_penalty": 1.2,
                    "num_ctx": 20000,
                    "min_p" : 0.1

                }
            )
            if 'tool_calls' in response['message']:
                for call in response['message']['tool_calls']:
                    #print(f"[bold green]Tool Call:[/][bold yellow] {call.function.name} [/][green]with arguments [/][bold yellow]{call.function.arguments}[/]")
                    try:
                        if call.function.name in [x.name for x in self.llamascope.get_tools()]:
                            print("[green]Reading code...[/] ⚙️               ", end="\r" )
                            result = self.llamascope.handle_tool_call(call)
                    except Exception as e:
                        messages+=[{'role':'tool', 'name': call.function.name, 'content': f'Error: tool {call.function.name}: {str(e)}'}]
                        continue
                    if not result:
                        result = ''
                    messages+=[{"role": "tool", "name": call.function.name, "content": result}]
                    #print(" Tool Call Output:")
                    #print(messages[-1])
                if len(messages) > 8:
                    if (messages[-1]['content'] == messages[-5]['content'] and
                        messages[-3]['content'] == messages[-7]['content']):
                        messages[-1] = {'role':'tool', 'name': call.function.name, 'content': """
                                  Hey, it looks like you have been calling the same API for a while.
                                  Are you sure this is actually the right place to look for the information you need?
                                  Rethink your life choices, agent. You have been warned.
                                  Now try again taking a different strategy.
                                  """}
                        print('\n😵😵😵                                      ')
            else:
                try:
                    # Unolad task module
                    #host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                    #url = f"{host}/api/delete"
                    #requests.delete(url, json={'name': model})
                    return json.loads(response['message']['content'].strip('```json\n'))
                except Exception as e:
                    raise ValueError(f"Invalid JSON returned by TaskAgent: {e} Raw content: {response['message']['content']}")
                break

    def call_prompt_agent(self, prompt: str, model="qwen2.5-coder:32b"):
        rules = self.get_rules('design')
        if len(rules) > 0:
            rules = '\nHere are some mandatory rules to follow:\n' + rules + '\n'
        system_prompt = """
        You are a C code parser and task planner.
        Your job is to transform the input into software specifications.
        To define the tasks, you are given access to tools via function-call tools
        format, which you can use to explore the codebase and to extract content from web pages on the internet.

        You never make up the final response until all the required fields are filled.
        You never assume the existence of any types, symbols, functions, files or macros.
        You are skeptical that those exist in the codebase at all, so you MUST take at every symbol you want to mention in your reply, and check if they exist in the codebase. look at these using the tool calls API provided.

        Do not write code, provide an action plan in natural language so that a team of expert code developers will execute.
        Only work to define development steps. Do not mention testing or deploying, unless specifically mentioned in the original prompt.

        Work one step at a time, as follows:
        - Read the prompt. Understand what the user actually meant based on your current knowledge of facts. Rephrase using the correct terminology for the context.
        - Research. Look for any useful information on the internet if you are not sure about the concepts mentioned, the meaning of any terminology used, or the mechanisms involved to reach the final goal.
        - Read the code. Look in the codebase to find any relevant symbols, types, macros, files, functions. Select all those that are relevant for the tasks and list them into separate categories that are required in the final answer.
        - Identify and list: Identify and list all the symbols (types, macros, files, functions) that will be affected by the action being proposed to the development team.
        - Loop: Repeat [Read-the-prompt, Research, Read-the-code, Identify-and-list] sequence until the lists are complete and the symbols are identified.
        - Final response: Only when all the required field are filled, I will give a final response in the bullet-list format required.


        """ + rules

        usr_prompt = """


        Your final response will only consists of a bullet list, as follows:
          - Discoveries: The things you have learned browsing the internet.
          - Actions: The actions that the development team will take. Maximum three sentences in natural language.
          - Types: a list of Relevant types (e.g. structs, unions, enums). Those are actual types defined in the code. Look them up.
          - Macros: a list of Relevant macros existing in the codebase. Those are also already in the code, find the relevant ones.
          - Files: a list of full paths to affected files in the codebase (can be existing or new). Always use full path. YOU MUST use full path. Only full paths are accepted by tools and patches.
          - Functions: a list of Relevant functions existing in the codebase (only existing ones you were able to look up in the codebase)
          - TODO: a list of names and descriptions of symbols to defines, e.g. types or functions to introduce and implement from scratch.

         Don't create the final response until you can fill all the requested fields. Use the tool calls API to identify all the items requested. Do not make up symbol or file names. Do not make assumptions on symbols existence until you checked with the provided API calls.

         """
        #usr_prompt += """
        #This is a list of the documents available:
        #""" + '\n'.join(DocReaderTool.available) + '\n' +
        usr_prompt += """
        This is the prompt from the user:
        """ + prompt

        tools = self.llamascope.get_tools() + self.web.get_tools() #+ self.doc.get_tools()
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.model_json_schema()
                }
            }
            for t in tools
        ]
        #print("Tools: " + ','.join([x['function']['name'] for x in tool_defs]))
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_prompt + '\n' + 'OUTPUT ACTION PLAN:\n'}
        ]
        #print(messages)
        while True:
            print("\n[cyan]Thinking...[/] 🤔💭 ", end='\r')
            response = ollama.chat(
                model=model,
                messages=messages,
                tools=tool_defs,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2,
                    "num_ctx": 20000
                }
            )
            if 'tool_calls' in response['message']:
                for call in response['message']['tool_calls']:
                    #print(f"[bold cyan]Tool Call:[/][bold yellow] {call.function.name} [/][green]with arguments [/][bold yellow]{call.function.arguments}[/]")
                    try:
                        if call.function.name in [x.name for x in self.llamascope.get_tools()]:
                            print("[cyan]Reading code...[/] ⚙️                    ", end = '\r')
                            result = self.llamascope.handle_tool_call(call)
                        elif call.function.name in [x.name for x in self.web.get_tools()]:
                            print("\n[green]Browsing the internet...[/] 🌐🌐🌐     ", end='\r')
                            result = self.web.handle_tool_call(call)
                            #print("Web result:\n", result)
                        elif call.function.name in [x.name for x in self.doc.get_tools()]:
                            print(f"\n[red]Reading books...[/]    📖🤓                ", end = '\r')
                            #print(f"[bold cyan]Tool Call:[/][bold yellow] {call.function.name} [/][green]with arguments [/][bold yellow]{call.function.arguments}[/]")
                            result = self.doc.handle_tool_call(call)
                    except Exception as e:
                        messages+=[{'role':'tool', 'name': call.function.name, 'content': f'Error: tool {call.function.name}: {str(e)}'}]
                        continue
                    if not result:
                        result = ''
                    messages+=[{"role": "tool", "name": call.function.name, "content": result}]
                    #print(" Tool Call Output:")
                    #print(messages[-1])
                if len(messages) > 8:
                    if (messages[-1]['content'] == messages[-5]['content'] and
                        messages[-3]['content'] == messages[-7]['content']):
                        messages[-1] = {'role':'tool', 'name': call.function.name, 'content': """
                                  Hey, it looks like you have been calling the same API for a while.
                                  Are you sure this is actually the right place to look for the information you need?
                                  Rethink your life choices, agent. You have been warned.
                                  Now try again taking a different strategy.
                                  """}
                        print('\n😵😵😵                                      ')
            else:
                return response['message']['content']

    def run(self, prompt):
        if '@research' in prompt:
            try:
                expanded_prompt = self.call_prompt_agent(prompt)
            except ValueError as e:
                print("Failed to parse prompt:", e)
                return
            expanded_prompt = 'User requested: ' + prompt + '\nI have expanded that request into the following actions:\n\n' + expanded_prompt
            print(f"[bold green]{expanded_prompt}[/bold green]")
        else:
            expanded_prompt = prompt
        patches = []
        error_retry = 0
        while len(patches) == 0:
            try:
                print('[green]Tasking...[/]  ✅                           ', end='\r' )
                tasks = self.call_task_agent(expanded_prompt)
            except ValueError as e:
                print("Failed to parse task list:", e)
                return
            for task in tasks:
                task_type = task.get("type")
                target = task.get("target")
                file = task.get("file")

                if not task_type or not target:
                    print("[red]Invalid task:[/]", task)
                    break

                print(f"Dispatching task: {task_type} for target '{target}': {task.get('details')}")

                try:
                    task_patches = self.dispatch_to_fim(task)
                    if len(task_patches) == 0:
                        task_patches = self.dispatch_to_editor(task)
                    if len(task_patches) == 0:
                        break
                except Exception as err:
                    task_patches = [ {'error':err} ]


                for patch in task_patches:
                    error = patch.get('error')
                    if error:
                        print('[red]Error in patch[/]', error)
                        break


                    adding = patch.get('adding', '')
                    removing = patch.get('removing', '')
                    add_lines = adding.count('\n')
                    remove_lines = removing.count('\n')
                    if len(adding) > 0 and not adding.endswith('\n'):
                        add_lines += 1
                    if len(removing) > 0 and not removing.endswith('\n'):
                        remove_lines += 1
                    if add_lines == 0 and remove_lines == 0:
                        print('[red]Empty patch[/]')
                        break
                    print(f"Patching {patch['path']}:{str(patch['line'])}:\n" + str(add_lines) + '+, ' + str(remove_lines) + '-\n')
                    self.llamascope.ghost_apply(patch)
                    patches.append(patch)
            if error:
                error = None
                error_retry += 1
                print("[red][FIM]Error executing pipeline[/]")
                if error_retry > 3:
                    break
                print("[red][FIM]Retrying.[/]")
                patches = []
                continue

        print("🩹 Total patches: 🩹" + str(len(patches)))
        response = { 'role': 'assistant', 'message': {'content': {'patches': patches} } }
        return response

    def dispatch_to_fim(self, task):
        file = task.get('file')
        if not file:
            print("[red][FIM]No file specified[/]")
            return None

        print(f"\n\n[bold green] 👻 [FIM]{task['type']}:[/] {task['file']}, {task['target']}. [bold green]{task['details']}[/]")

        deleting = ""
        prefix = ''
        suffix = ''
        context = ''

        if task['type'] in ('FunctionGeneration', 'SymbolImplementation'):
            file_path = task.get('file')
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(f'/* {file_path} */ ')
            with open(file_path, 'r') as f:
                content = f.read()
                line = content.count('\n')
            context = ''
            context += 'Generate a new function called ' + task['target'] + '\n'
            file = file_path

        elif task['type'] == 'FunctionRefactor' or task['type'] == 'StubCompletion':
            sym = self.llamascope.get_symbol(task['target'])
            if not sym:
                print(f'[red]Error: Symbol {task["target"]} not found.[/red]')
                return []
            file = sym['file']
            line = sym['line']
            with open(file, 'r') as f:
                code = f.read()
            deleting = code[sym['start']:sym['end']]
            prefix = '/* Original function before the requested changes: */ \n' + deleting + '\n'
            prefix += '/ * Rules:\n'
            prefix += '  * - Do not alter the function signature or argument names.\n'
            prefix += '  * - Do not alter parts of the function you are not explicitly requested to modify\n'
            prefix += '  * - Always complete the code. DO NOT leave stubs or unfinished, non-working code.\n'
            prefix += '  * - If parts of the original code should be part of the answer, copy them rather than referencing the removed code.\n'
            prefix += '  */'
            prefix += self.llamascope.get_called_functions(task['target']) + '\n\n'
            suffix = self.llamascope.get_callers(task['target']) + suffix

        elif task['type'] in ('InlineFix', 'MiniRewrite'):
            sym = self.llamascope.get_symbol(task['target'])
            if not sym:
                print(f'[red]Error: Symbol {task["target"]} not found.[/red]')
                return []
            line = sym['line']
            file = sym['file']
            prefix += self.llamascope.get_called_functions(task['target']) + '\n\n'
            suffix = self.llamascope.get_callers(task['target']) + suffix
        elif task['type'] == 'TypeDefinition':
            file_path = task.get('file')
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(f'/* {file_path} */ ')
            with open(file_path, 'r') as f:
                content = f.read()
                line = content.count('\n')
            context = ''
            context += 'Generate a new type called ' + task['target'] + '\n'
            file = file_path
            with open(file_path, 'rb') as f:
                src = f.read().decode('utf-8', errors='replace')
                line = src.count('\n')
        elif task['type'] == 'MacroDefinition':
            sym = self.llamascope.get_symbol(task['target'])
            file = task.get('file')
            if not sym:
                context = f'The new macro`{task["target"]}` is not yet defined in the current file. Write it from scratch.'
                with open(file, 'rb') as f:
                    content = f.read().decode('utf-8', errors='replace')
                if not '#define' in content:
                    line = 0
                else:
                    idx = content.index('#define')
                    line = content[0:idx].count('\n') - 1
            else:
                line = sym['line']
                with open(file, 'rb') as f:
                    code = f.read().decode('utf-8', errors='replace')
                deleting = code[sym['start']:sym['end']]
            prefix = '/* Original macro before the requested changes: */ \n' + deleting + '\n'
            context += 'Rules:\n'
            context += '- If parts of the original code should be part of the answer, copy them rather than referencing the removed code.\n'
        else:
            return []

        response_insert = self.call_fim(prefix, suffix, context, file, line, task)

        if response_insert:
            if (deleting and deleting.count('\n') > 0):
                print('[yellow]Deleting replaced code[/]')
                response_insert['removing'] += deleting
            code = response_insert['adding']
            tree = self.llamascope.parser.parse(code.encode())
            node = tree.root_node.children[0]
            code = code[0:node.end_byte] # Include from the beginning to avoid losing code comments
            if not code.endswith('\n'):
                code += '\n'
            response_insert['adding'] = code
        return [response_insert]

    def call_fim(self, prefix: str, suffix: str, context: str, file: str, line: int, task: dict) -> str:
        model = "qwen2.5-coder:14b"
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        url = f"{host}/api/generate"
        # Sanitize and format context
        sanitized_context = context.replace('/*', '').replace('*/', '').replace('//', '')
        rules = self.get_rules('coding')
        if len(rules) > 0:
            rules = '\nHere are some mandatory rules to follow:\n' + rules + '\n'

        fim_comment_block = f"/*\nContext:\n{sanitized_context}\n\nGoal:\n{task['details']}\n\n"
        fim_comment_block += rules
        fim_comment_block += '\n*/\n'
        prefix += self.llamascope.get_types(task['ADTs']) +'\n\n'
        prefix += self.llamascope.get_references(task['references']) + '\n\n'
        result = ''
        complete = False
        found_function = False
        print(f'[bold yellow] 👻 Crafting Code 👻: {file}:{str(line)}[/]')
        print(f'[yellow]📎 Context len: prefix {str(len(prefix))} B, suffix {str(len(suffix))} B[/]')

        # VERBOSE
        #print(f'prefix:\n {prefix}')

        while not complete:
            # Construct Qwen2.5 FIM prompt
            fim_prompt = f"<|fim_prefix|>{prefix}\n{fim_comment_block}<|fim_suffix|>{suffix}<|fim_middle|>"
            payload = {
                "model": model,
                "prompt": fim_prompt,
                "options": {
                    "temperature": 0.05,
                    "top_p": 0.8,
                    "repeat_penalty": 1.1,
                    "num_ctx": 12000,
                    "num_predict":8192,
                    "min_p": 0.2
                },
                "stream": False,
                "raw": True
            }
            #print(f'[magenta]{fim_comment_block}\n[/]')
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                result += response.json().get("response", "")
                if '```' in result:
                    result = result.split('```')[0]
                print("[yellow]Received FIM response...[/]")
            except requests.RequestException as e:
                print(f"[FIM Error] {e}")
                return []

            # If there is an incomplete function in the generation, call again
            if not complete:
                try:
                    tree = ExtendedLlamaScope.parser.parse(result.encode())
                    root = tree.root_node
                except Exception as e:
                    print(f"[bold red][Parser Error][/] {e}")

                for node in root.children:
                    if node.type == 'function_definition':
                        found_function = True
                        # Check if this function node spans the entire input
                        if node.end_byte <= len(result.encode()):
                            print("[bold yellow]FIM reply function complete[/]")
                            complete = True
                            result = result.encode()[node.start_byte:node.end_byte].decode(errors='replace')
                        break
                if not found_function:
                    print("[bold red]FIM reply no function found[/]")
                    complete = True
            if not complete:
                prefix += result

        try:
            patch = { "path" : file,
                    "line" : line,
                    "adding" : result.strip(),
                    "removing" : ""
                     }

        except Exception as e:
            print(f"[bold red][FIM Error][/] {e}")
        return patch


    def dispatch_to_editor(self, task):
        print(f"[Editor] Processing: {task['type']} for target {task['target']}")
        code = ""
        file = task['file']
        line = -1
        matches = []
        if task['type'] == 'SymbolRename':
            oldname = task['target']
            newname = task['target_new'].split(' ')[-1]
            print("[Editor] Renaming symbol from:", oldname, "to", newname)
            with open(file, 'rb') as f:
                src = f.read()
            if oldname.encode() not in src:
                print('no matches.\n')
                return []
            lines = src.decode(errors='replace').splitlines()
            byte_to_line = {}
            offset = 0
            tree = self.llamascope.parser.parse(src)

            for i, line in enumerate(lines):
                for j in range(len(line) + 1):
                    byte_to_line[offset + j] = i
                offset += len(line) + 1

            def symbol_rename_visit(node):
                #if node.type == "identifier" or node.type == "type_identifier" or node.type == "function_declarator" or node.type == "function_definition":
                name = src[node.start_byte:node.end_byte].decode(errors='replace')
                if name.count('\n') == 0 and oldname in name:
                    #print(f'Node type: {node.type} Name: {name} Start byte: {node.start_byte} End byte: {node.end_byte}')
                    matches.append(node)
                else:
                    for child in node.children:
                        symbol_rename_visit(child)
            symbol_rename_visit(tree.root_node)

            patches = []
            for n in reversed(matches):
                start = n.start_byte
                end = n.end_byte
                line_num = byte_to_line.get(start, -1)
                if line_num < 0 or line_num >= len(lines):
                    continue

                print(f'At line n. {str(line_num)}')
                orig_line = lines[line_num]
                print(f'Original line: {orig_line}')


                # Compute start/end column
                byte_start_of_line = sum(len(l) + 1 for l in lines[:line_num])
                start_col = start - byte_start_of_line
                end_col = end - byte_start_of_line

                new_line = orig_line[:start_col] + orig_line[start_col:end_col].replace(oldname, newname) + orig_line[end_col:]
                print(f'New line: {new_line}')
                patch = {'path': file,
                         'line': line_num + 1,
                         'adding': new_line,
                         'removing': orig_line
                }
                patches.append(patch)
            return patches
        elif task['type'] == 'FileMove':
            with open(task['file'], 'rb') as f:
                code = f.read().decode('utf-8', errors='replace')
            patches = [
                    { 'path' : task['file'],
                      'line' : 0,
                      'adding': '',
                      'removing': code
                    },
                    { 'path' : task['target'],
                      'line' : 0,
                      'adding' : code,
                      'removing' : ''
                    }
                ]
            return patches
        elif task['type'] == 'IncludeFix':
            if not '#include' in task['target']:
                return [{"error": "invalid 'IncludeFix' target: " + task['target' ]}]
            file = task['file']
            with open(file, 'rb') as f:
                code = f.read().decode('utf-8', errors='replace')
            if '#include' in code:
                idx = code.index('#include')
                line = code[:idx].count('\n')
            else:
                line = 0

            patch = {
                    "path" : file,
                    "line" : line,
                    "adding": task['target'] + '\n',
                    "removing": ""
                }
            return [patch]
        elif task['type'] == 'DeleteFunction':
            sym = self.llamascope.get_symbol(task['target'])
            print(f"[Editor] Deleting function: {sym['name']}")
            print(f"[Editor] From file: {sym['file']} line: {str(sym['line'])}")
            file = sym['file']
            line = sym['line']
            with open(file, 'r') as f:
                code = f.read()[sym['start']:sym['end']]
            patch = {"path" : file,
                     "line" : line,
                     "adding": "",
                     "removing": code
                    }
            return [patch]
        else:
            return [{"error": "invalid task for [Editor]: " + task['type']}]

def convert_patches_to_unified_diff(patches: list[dict]) -> str:
    """Convert a list of patches in JSON format into standard unified diff format."""
    diff_lines = []

    for patch in patches:
        file = patch.get("path", "unknown.c")
        line = patch.get("line", 0)
        added = patch.get("adding", "").splitlines()
        deleted = patch.get("removing", "").splitlines()

        diff_lines.append(f"--- {file}")
        diff_lines.append(f"+++ {file}")
        diff_lines.append(f"@@ -{line},{len(deleted)} +{line},{len(added)} @@")

        for old in deleted:
            diff_lines.append(f"-{old}")
        for new in added:
            diff_lines.append(f"+{new}")

        # blank line between diffs
        diff_lines.append("")
    return "\n".join(diff_lines)



class Ghost:
    def __init__(self, path):
        try:
            os.mkdir(path)
        except:
            pass
        self.ws = Workspace(container=Container(path))
        self.pipeline = PipeLine(self.ws, path)

    def invoke(self, prompt):
        self.pipeline.run(prompt)

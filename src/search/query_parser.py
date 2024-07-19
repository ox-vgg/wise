import json
import copy

def which_keyword_is_next(cmd, start_index, keyword_list, ignore_case=True):
    for ki in range(0, len(keyword_list)):
        key = keyword_list[ki]
        n = len(key)
        nchars = cmd[start_index:start_index+n]
        if ignore_case:
            key = key.lower()
            nchars = nchars.lower()
        if key == nchars:
            return ki
    return -1

def parse_search_query(cmd):
    """ Generate a parse tree from search query

    Parameters
    ----------
    cmd : the search query command

    Returns
    -------
    dict : search query represented as a tree of atomic operations


    Example search queries:
    wash hands in video but not wash hands in metadata
    cooking in video and background music in audio
    cooking in video and music in audio and singing not in metadata
    music in audio and singing not in metadata
    music in audio or singing in metadata or music in metadata or noise not in metadata
    """
    SEARCH_TARGET_LIST = ['audio', 'video', 'metadata']
    SEARCH_TARGET_LINK = ['in', 'not in']
    QUERY_RESULT_MERGE_OPS = ['and', 'or']
    QUERY_FILE_PREFIX = '@'
    QUERY_QUOTE_CHAR = ['"', "'"]
    QUERY_EMBEDDING_VEC_OPS = ['+', '-']

    status = 'ERROR'
    message = ''
    parsed_queries = {
        'cmd': cmd,                    # original search query string
        'query': [],                   # set of sub-queries
        'query_result_merge_ops':[]    # AND, OR, ...
    }

    # state
    query_extract_done = False
    query_spec_template = {
        'query_str': [],               # query string or @filename
        'query_type': [],              # {'file', 'text'}
        'query_embedding_vec_op': [],  # {'+', '-'}
        'search_target': '',           # {audio, video, metadata}
        'search_target_link': ''       # {'in', 'not in'}
    }
    parsed_queries['query'].append(copy.deepcopy(query_spec_template))
    query_index = 0

    index = 0
    while index < len(cmd):
        # Step 1: parse query (text or @filename)
        if not query_extract_done:
            if cmd[index] == QUERY_FILE_PREFIX:
                query_type = 'file'
                index += 1
            else:
                query_type = 'text'
            if cmd[index] in QUERY_QUOTE_CHAR:
                quote_begin = index
                quote_char = cmd[quote_begin]
                quote_end = cmd.find(quote_char, quote_begin + 1)
                if quote_end == -1:
                    message = f'expected close of quotation character [{quote_char}] started at {quote_begin}'
                    break
                query_str = cmd[quote_begin:quote_end+1]
                next_char = quote_end + 1
                if cmd[next_char] != ' ':
                    message = f'expected a space character after quoted query {query_str}'
                    break
                index = next_char + 1
            else:
                next_char_index = index
                query_str_end = -1
                while next_char_index < len(cmd):
                    next_space = cmd.find(' ', next_char_index)
                    if next_space == -1:
                        message = f'expected a space character after index {index}'
                        break
                    next_char_index = next_space + 1
                    if cmd[next_char_index] in QUERY_EMBEDDING_VEC_OPS:
                        query_str_end = next_space
                        break
                    target_link_index = which_keyword_is_next(cmd, next_char_index, SEARCH_TARGET_LINK)
                    if target_link_index != -1:
                        query_str_end = next_space
                        break
                    query_char_index = next_char_index
                if query_str_end == -1:
                    message = f'unable to locate query'
                    break
                query_str_begin = index
                query_str = cmd[ query_str_begin:query_str_end ]
                index = query_str_end + 1
            parsed_queries['query'][query_index]['query_str'].append(query_str)
            parsed_queries['query'][query_index]['query_type'].append(query_type)
            query_extract_done = True

        # Step 2:
        # Step 2.1 parse embedding vector operator (+ or -)
        # Step 2.2 parse search target link flags (i.e. NOT IN or IN)
        # Step 2.3 parse search target (e.g. audio, video, metadata)
        # Step 2.4 parse sub-query result merge operator (e.g. AND, OR)
        if query_extract_done:
            # Step 2.1 parse embedding vector operator (+ or -)
            if cmd[index] in QUERY_EMBEDDING_VEC_OPS:
                # vector operator (+ or -) seems to have been provided
                if cmd[index + 1] == ' ':
                    parsed_queries['query'][query_index]['query_embedding_vec_op'].append( cmd[index] )
                    index += 2
                    query_extract_done = False
                    continue  # continue to extract query-{text,file}
                else:
                    message = f'expected a space character after query embedding vector operator (+,-)'
                    break

            # Step 2.2 parse search target link flags (i.e. NOT IN or IN)
            target_link_index = which_keyword_is_next(cmd, index, SEARCH_TARGET_LINK)
            if target_link_index == -1:
                message = f'expected IN or NOT IN after query {parsed_queries["query"][query_index]["query_str"]}'
                break
            search_target_link = SEARCH_TARGET_LINK[target_link_index]
            parsed_queries['query'][query_index]['search_target_link'] = search_target_link
            index += len(search_target_link) + 1

            # Step 2.3 parse search target (e.g. audio, video, metadata)
            target_index = which_keyword_is_next(cmd, index, SEARCH_TARGET_LIST)
            if target_index == -1:
                message = f'expected one of {SEARCH_TARGET_LIST} but got "{cmd[index:index+10]}"'
                break
            search_target = SEARCH_TARGET_LIST[target_index]
            parsed_queries['query'][query_index]['search_target'] = search_target
            index += len(search_target)

            if index == len(cmd):
                status = 'OK'
                message = f'parsed {len(parsed_queries["query"])} sub-queries'
                break

            if cmd[index] != ' ':
                message = f'expected a space character before one of merge operation from {QUERY_RESULT_MERGE_OPS}'
                break

            # Step 2.4 parse sub-query result merge operator (e.g. AND, OR)
            index += 1
            merge_op_index = which_keyword_is_next(cmd, index, QUERY_RESULT_MERGE_OPS)
            if merge_op_index == -1:
                message = f'expected a result merge operator from {QUERY_RESULT_MERGE_OPS}'
                break
            merge_op = QUERY_RESULT_MERGE_OPS[merge_op_index]
            parsed_queries['query_result_merge_ops'].append(merge_op)
            index += len(merge_op)
            if cmd[index] != ' ':
                message = f'expected a space character after merge operation "{merge_op}" and before start of the next query'
                break
            index += 1

            # get ready to parse remaining queries
            query_extract_done = False
            parsed_queries['query'].append(copy.deepcopy(query_spec_template)) # space for new sub-queries
            query_index = query_index + 1

    query_parser_status = {
        'status': status,
        'message': message
    }
    return query_parser_status, parsed_queries

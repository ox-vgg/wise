# Create a LISA manual annotation project based on search results
# for a set of queries to the Wikimedia Commons' metadata based
# search engine as well as the WISE search engine for Wikimedia
# Commons' images
#
# Author: Abhishek Dutta <adutta@robots.ox.ac.uk>
# Date  : 2023-03-27

import json
import uuid
import time
from datetime import datetime
import argparse
import os

import hashlib
from urllib.parse import urlencode, quote_plus, quote
from urllib.request import urlopen
import requests
import http.client

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f') # save float with only 3 decimal places

search_query_list = [
    'toy aeroplane',
    'pencil drawing of lion',
    'a train in mountains near waterfall',
    'cheetah running',
    'hot air balloon above a mountain',
    'dolphin playing with ball',
    'penguin with wings raised',
    'running on a hill',
    'people on a roller coaster',
    'car with a bicycle on top'
]

class LISA():
    """List Annotator (LISA) project, see https://gitlab.com/vgg/lisa"""
    def __init__(self, project_name=None):
        self._fid_to_findex = {}
        self._filename_to_fid = {}
        self._lisa = {}
        project_id = str(uuid.uuid4())
        now_timestamp = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z'))
        self._lisa['project'] = {
            'shared_fid':'__FILE_ID__',
            'shared_rev':'__FILE_REV_ID__',
            'shared_rev_timestamp':'__FILE_REV_TIMESTAMP__',
            'project_name':project_name,
            'project_id': project_id,
            'creator':'List Annotator - LISA (https://gitlab.com/vgg/lisa)',
            'editor':'List Annotator - LISA (https://gitlab.com/vgg/lisa)',
            'file_format_version':'0.0.3',
            'created_timestamp':now_timestamp,
        }
        self._lisa['config'] = {
            "file_src_prefix": "",
            "navigation_from": 0,
            "navigation_to": 256,
            "item_per_page": 256,
            "float_precision": 4,
            "item_height_in_pixel": 300,
            'show_attributes': { 'file':[], 'region':[] }
        }

        self._lisa['attributes'] = {}
        self._lisa['attributes']['file'] = {
            'width': {'aname':'Width', 'atype':'text'},
            'height': {'aname':'Height', 'atype':'text'},
        }

        self._lisa['attributes']['region'] = {}
        self._lisa['files'] = []

    def config(self, key, value):
        self._lisa['config'][key] = value

    def set_project_editor(self):
        self._lisa['project']['editor'] = ''

    def add_attribute(self, atype, attribute_id, attribute_def):
        if atype != 'file' and atype != 'region':
            print('attribute atype must be {file, region}')
            return
        if attribute_id in self._lisa['attributes'][atype]:
            print('attribute %s already exists' % (attribute_id))
            return
        self._lisa['attributes'][atype][attribute_id] = attribute_def
        self._lisa['config']['show_attributes'][atype].append(attribute_id)

    def add_file(self, filename, width, height):
        findex = len(self._lisa['files'])
        self._lisa['files'].append( {
            'fid':findex,
            'src':filename,
            'regions':[],
            'rdata':[],
            'fdata': {
                'width':width,
                'height':height,
            }
        })
        self._filename_to_fid[filename] = findex
        return findex

    def add_region(self, findex, region, rdata):
        rindex = len(self._lisa['files'][findex]['regions'])
        self._lisa['files'][findex]['regions'].append(region)
        self._lisa['files'][findex]['rdata'].append(rdata)
        return rindex

    def load_json(self, fn):
        with open(fn, 'r') as f:
            self._lisa = json.load(f)
            print('Loaded LISA project from %s' % (fn))

    def save_json(self, fn):
        with open(fn, 'w') as f:
            json.dump(self._lisa, f)

def search_wikimedia(base_url, query, result_count=50):
    query = {
        'action':'query',
        'list':'search',
        'srprop':'sectiontitle|isfilematch',
        'format':'json',
        'srlimit':str(result_count),
        'srnamespace':'6',
        'srsearch': query
    }
    query_url = base_url + urlencode(query, quote_via=quote_plus)
    http_response = urlopen(query_url)
    search_result = json.load(http_response)
    return search_result

def search_wise(base_url, username, password, query, result_count=50):
    query = {
        'start':0,
        'end':result_count,
        'q': query,
        'thumbs':0
    }
    query_url = base_url + urlencode(query, quote_via=quote_plus)
    if username == None and password == None:
        http_response = requests.get(query_url)
    else:
        http_response = requests.get(query_url, auth=(username, password))
    search_result = json.loads(http_response.text)
    return search_result

def create_shared_project(project_data):
	project_data_str = json.dumps(project_data, indent=None, separators=(',',':'))
	conn = http.client.HTTPSConnection('meru.robots.ox.ac.uk', 443)
	conn.request('POST', '/store/', project_data_str)
	response = conn.getresponse()
	shared_project_info = ''
	if response.status == 200:
		shared_project_info = response.read().decode('utf-8')
	else:
		via_project_id = 'ERROR:' + response.reason
	conn.close()
	return shared_project_info

def main():
    parser = argparse.ArgumentParser(description="Create manual annotation dataset to benchmark image retrieval performance")
    parser.add_argument("--out-lisa-dir",
                        required=True,
                        type=str,
                        help="all output LISA project JSON filename")
    parser.add_argument("--wikimedia-server-url",
                        required=False,
                        default="https://commons.wikimedia.org/w/api.php?",
                        type=str,
                        help="URL of Wikimedia Commons api server")
    parser.add_argument("--wise-server-url",
                        required=True,
                        type=str,
                        help="URL of WISE Image Search Engine (WISE) server (e.g. https://meru.robots.ox.ac.uk/wikimedia/)")
    parser.add_argument("--wise-username",
                        required=False,
                        default=None,
                        type=str,
                        help="credentials if WISE requires username/password")
    parser.add_argument("--wise-password",
                        required=False,
                        default=None,
                        type=str,
                        help="credentials if WISE requires username/password")

    args = parser.parse_args()

    if not os.path.exists(args.out_lisa_dir):
        os.makedirs(args.out_lisa_dir)
    wikimedia_dir = os.path.join(args.out_lisa_dir, 'wikimedia')
    wise_dir = os.path.join(args.out_lisa_dir, 'wise-wikimedia5-35M')
    if not os.path.exists(wikimedia_dir):
        os.makedirs(wikimedia_dir)
    if not os.path.exists(wise_dir):
        os.makedirs(wise_dir)
    shared_pid_summary_fn = os.path.join(args.out_lisa_dir, 'shared-pid-list.csv')
    shared_pid_summary_f = open(shared_pid_summary_fn, 'a')

    file_attributes = {
        'rank': {
            'aname':'Rank',
            'atype':'label'
        },
        'infopage': {
            'aname':'Info',
            'atype':'label'
        },
        'filename': {
            'aname':'Filename',
            'atype':'label'
        },
        'source': {
            'aname':'Source',
            'atype':'label'
        },
        'is_correct': {
            'aname':'Is Correct?',
            'atype':'radio',
            'options': { 'y':'Yes', 'n':'No', 'na':'Cannot Say' }
        }
    }

    wikimedia_base_url = args.wikimedia_server_url
    wise_base_url = args.wise_server_url

    visible_file_attribute_list = ['rank', 'infopage', 'source', 'is_correct']
    RESULTS_TO_RETURN = 100
    CSV_HEADER = '# search_query,wikimedia-lisa-shared-pid,wise-lisa-shared-pid'
    print(CSV_HEADER)
    shared_pid_summary_f.write(CSV_HEADER)
    for qi in range(0, len(search_query_list)):
        search_query = search_query_list[qi]
        search_query_santized = search_query.replace(' ', '-')
        print('%s' % (search_query), end='')

        ## Wikimedia Commons
        wikimedia_lisa = LISA(search_query)
        wikimedia_lisa._lisa['attributes']['file'] = file_attributes
        wikimedia_lisa._lisa['config']['show_attributes']['file'] = visible_file_attribute_list
        wikimedia_result = search_wikimedia(wikimedia_base_url, search_query, RESULTS_TO_RETURN)
        wikimedia_file_id = 0
        for ri in range(0, len(wikimedia_result['query']['search'])):
            title = wikimedia_result['query']['search'][ri]['title']
            title_tok = title.split('File:')
            if len(title_tok) != 2:
                print('Invalid title: [%s' % (title))
                continue
            filename = title_tok[1].replace(' ', '_')
            filename_md5 = hashlib.md5( filename.encode('utf-8') ).hexdigest()
            wikimedia_file_page = 'https://commons.wikimedia.org/wiki/File:' + filename
            file_url = 'https://upload.wikimedia.org/wikipedia/commons/' + filename_md5[0] + '/' + filename_md5[0:2] + '/' + filename
            wikimedia_lisa._lisa['files'].append( {
                'fid':wikimedia_file_id,
                'src':file_url,
                'regions':[],
                'rdata':[],
                'fdata': {
                    'infopage': '<a target="_blank" href="' + wikimedia_file_page + '">doi</a>',
                    'filename': filename,
                    'source': 'Wikimedia Commons',
                    'query': search_query,
                    'rank':ri
                }
            })
            wikimedia_file_id = wikimedia_file_id + 1
        wikimedia_lisa._lisa['config']['item_per_page'] = RESULTS_TO_RETURN
        wikimedia_lisa_fn = os.path.join(wikimedia_dir, search_query_santized + '.json')
        wikimedia_lisa.save_json(wikimedia_lisa_fn)
        wikimedia_shared_pinfo_str = create_shared_project(wikimedia_lisa._lisa)
        wikimedia_shared_pinfo = json.loads(wikimedia_shared_pinfo_str)
        print(',%s' % (wikimedia_shared_pinfo['shared_fid']), end='')

        ## WISE
        wise_lisa = LISA(search_query)
        wise_lisa._lisa['attributes']['file'] = file_attributes
        wise_lisa._lisa['config']['show_attributes']['file'] = visible_file_attribute_list
        wise_result = search_wise(wise_server_url, args.wise_username, args.wise_password, search_query, RESULTS_TO_RETURN)
        wise_result = wise_result[search_query]
        wise_file_id = 0
        for ri in range(0, len(wise_result)):
            img_link = wise_result[ri]['link'];
            img_link_tok = img_link.split('/');
            img_filename = img_link_tok[ len(img_link_tok) - 2 ];
            wikimedia_file_page = 'https://commons.wikimedia.org/wiki/File:' + img_filename
            wise_lisa._lisa['files'].append( {
                'fid':wise_file_id,
                'src':wise_result[ri]['link'],
                'regions':[],
                'rdata':[],
                'fdata': {
                    'infopage': '<a target="_blank" href="' + wikimedia_file_page + '">doi</a>',
                    'filename': img_filename,
                    'source': 'WISE',
                    'query': search_query,
                    'rank': ri,
                    'is_correct':'y'
                }
            })
            wise_file_id = wise_file_id + 1
        wise_lisa._lisa['config']['item_per_page'] = RESULTS_TO_RETURN
        wise_lisa_fn = os.path.join(wise_dir, search_query_santized + '.json')
        wise_lisa.save_json(wise_lisa_fn)
        wise_shared_pinfo_str = create_shared_project(wise_lisa._lisa)
        wise_shared_pinfo = json.loads(wise_shared_pinfo_str)
        print(',%s' % (wikimedia_shared_pinfo['shared_fid']), end='\n')
        shared_pid_summary_f.write('\n%s,%s,%s' % (search_query, wikimedia_shared_pinfo['shared_fid'], wise_shared_pinfo['shared_fid']))
    shared_pid_summary_f.close()

if __name__ == '__main__':
    main()

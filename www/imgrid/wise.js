/*
JS code for driving imgrid theme of WISE Image Search Engine

Author : Abhishek Dutta <adutta@robots.ox.ac.uk>
Date   : 2023-03-13
*/

const wise_data = {};
const PAGE_IMG_COUNT = 30;
const SEARCH_RESULT_COUNT = 20;
const MAX_SEARCH_RESULT = 1000;

// html containers
const imgrid_container = document.getElementById('imgrid');
const toolbar = document.getElementById('toolbar');
const navinfo1 = document.getElementById('navinfo1');
const navinfo2 = document.getElementById('navinfo2');

const UI_MODE = {
	'BROWSE_IMAGES': 'browse_images',
	'SHOW_RESULTS': 'show_results'
}

// state
var wise_home_rand_img_index_list = [];
var wise_home_from_findex = -1;
var wise_home_to_findex = -1;
var wise_result_start_findex = -1;
var wise_result_end_findex = -1;
var wise_current_ui_mode = UI_MODE.BROWSE_IMAGES;
var wise_total_page_count = -1;

//
// Home Page
//
const TOOLBAR_INFO = 'Try searching, for example, using <span class="text_button" onclick="search_for(\'bees feeding on flower\')">bees feeding on flower</span> or <span class="text_button" onclick="search_for(\'car on empty street in snow\')">car on empty street in snow</span> or <span class="text_button" onclick="search_for(\'horse in river\')">horse in river</span>.'
function init_home_page() {
	load_featured_image_grid();
	document.getElementById('toolbar').innerHTML = TOOLBAR_INFO;
}

function load_featured_image_grid() {
	wise_current_ui_mode = UI_MODE.BROWSE_IMAGES;

	fetch("info", {
		method: 'GET'
	})
		.then((response) => response.json())
		.then((project_info) => {
			wise_data['info'] = project_info;

			// initialise the toolbar with info
			const img_count_percent = ((wise_data['info']['num_images'] / 80000000) * 100).toFixed(0);
			document.getElementById('toolbar').innerHTML = 'Here, you can search nearly ' + img_count_percent + '% of the 80 million images in Wikimedia Commons.' + TOOLBAR_INFO;

			// get total page count
			navinfo1.innerHTML = 'Showing selected images.';
			navinfo2.innerHTML = 'Use arrow keys (or buttons) to navigate ' + wise_data['info']['num_images'].toLocaleString('en', { useGrouping: true }) + ' images.';
			imgrid_container.innerHTML = '';
			wise_home_rand_img_index_list = [];
			for (var i = 0; i < PAGE_IMG_COUNT; ++i) {
				rand_img_index = get_rand_int(0, wise_data['info']['num_images']);
				const img = document.createElement('img');
				img.src = 'thumbs/' + rand_img_index;
				const a = document.createElement('a');
				a.setAttribute('href', 'images/' + rand_img_index);
				a.setAttribute('target', '_blank');
				a.appendChild(img);

				wise_home_rand_img_index_list.push(rand_img_index)
				imgrid_container.appendChild(a);
			}
		});
}

// source: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random
function get_rand_int(min, max) {
	min = Math.ceil(min);
	max = Math.floor(max);
	return Math.floor(Math.random() * (max - min + 1) + min); // The maximum is inclusive and the minimum is inclusive
}

function show_next_page() {
	if (wise_current_ui_mode === UI_MODE.BROWSE_IMAGES) {
		if (wise_home_from_findex === -1 && wise_home_to_findex === -1) {
			wise_home_from_findex = 0;
			wise_home_to_findex = PAGE_IMG_COUNT;
		} else {
			wise_home_from_findex = wise_home_to_findex;
			wise_home_to_findex = wise_home_to_findex + PAGE_IMG_COUNT;
		}
		if (wise_home_to_findex > wise_data['info']['num_images']) {
			wise_home_from_findex = 0;
			wise_home_to_findex = PAGE_IMG_COUNT;
		}
		show_images(wise_home_from_findex, wise_home_to_findex)
	} else {
		if (wise_result_start_findex < MAX_SEARCH_RESULT && wise_result_end_findex < MAX_SEARCH_RESULT) {
			wise_result_start_findex = wise_result_end_findex;
			wise_result_end_findex = wise_result_end_findex + SEARCH_RESULT_COUNT;
			if (wise_result_end_findex >= MAX_SEARCH_RESULT) {
				wise_result_end_findex = MAX_SEARCH_RESULT;
			}
			subsequent_search_query(wise_result_start_findex, wise_result_end_findex);
		} else {
			toolbar.innerHTML = 'No more search results are available';
		}
	}
}

function show_prev_page() {
	if (wise_current_ui_mode === UI_MODE.BROWSE_IMAGES) {
		if (wise_home_from_findex === -1 && wise_home_to_findex === -1) {
			wise_home_to_findex = wise_data['info']['num_images'];
			wise_home_from_findex = wise_data['info']['num_images'] - PAGE_IMG_COUNT;
		} else {
			wise_home_to_findex = wise_home_from_findex;
			wise_home_from_findex = wise_home_from_findex - PAGE_IMG_COUNT;
		}
		if (wise_home_from_findex < 0) {
			wise_home_from_findex = wise_data['info']['num_images'] - PAGE_IMG_COUNT;
			wise_home_to_findex = wise_data['info']['num_images'];
		}
		show_images(wise_home_from_findex, wise_home_to_findex)
	} else {
		if (wise_result_start_findex > 0) {
			wise_result_end_findex = wise_result_start_findex;
			wise_result_start_findex = wise_result_start_findex - SEARCH_RESULT_COUNT;
			if (wise_result_start_findex < 0) {
				wise_result_start_findex = 0;
			}
			subsequent_search_query(wise_result_start_findex, wise_result_end_findex);
		} else {
			toolbar.innerHTML = 'Already in the first page of the search results!';
		}
	}
}

function show_images(from_findex, to_findex) {
	navinfo1.innerHTML = 'Showing images';
	imgrid_container.innerHTML = '';
	for (var i = from_findex; i < to_findex; ++i) {
		const img = document.createElement('img');
		img.src = 'thumbs/' + i;
		const a = document.createElement('a');
		a.setAttribute('href', 'images/' + i);
		a.setAttribute('target', '_blank');
		a.appendChild(img);

		imgrid_container.appendChild(a);
	}
	navinfo2.innerHTML = 'from ' + from_findex + ' to ' + to_findex + ' of ' + wise_data['info']['num_images'].toLocaleString('en', { useGrouping: true }) + ' images.'
}

//
// Search query and results
//
function search_for(query) {
	document.getElementById('search_keyword').value = query;
	submit_initial_search_query();
}

function submit_initial_search_query() {
	const search_keyword = document.getElementById('search_keyword').value;
	wise_result_start_findex = 0;
	wise_result_end_findex = SEARCH_RESULT_COUNT;
	const search_endpoint = 'search?q=' + search_keyword + '&start=' + wise_result_start_findex + '&end=' + wise_result_end_findex;

	toolbar.innerHTML = 'Searching for <strong>' + search_keyword + '</strong> in ' + wise_data['info']['num_images'].toLocaleString('en', { useGrouping: true }) + ' Wikimedia images <div class="spinner"></div>';

	const time0 = performance.now();
	fetch(search_endpoint, {
		method: 'GET'
	})
		.then((response) => response.json())
		.then((search_result) => {
			const time1 = performance.now();
			const search_time = time1 - time0;
			show_search_result(search_result, search_time);
		});
}

function subsequent_search_query(start, end) {
	const search_keyword = document.getElementById('search_keyword').value;
	const search_endpoint = 'search?q=' + search_keyword + '&start=' + start + '&end=' + end;

	toolbar.innerHTML = 'Continuing search for <strong>' + search_keyword + '</strong> in ' + wise_data['info']['num_images'].toLocaleString('en', { useGrouping: true }) + ' Wikimedia images <div class="spinner"></div>';

	const time0 = performance.now();
	fetch(search_endpoint, {
		method: 'GET'
	})
		.then((response) => response.json())
		.then((search_result) => {
			const time1 = performance.now();
			const search_time = time1 - time0;
			show_search_result(search_result, search_time);
		});

}

function show_search_result(response, search_time) {
	imgrid_container.innerHTML = '';
	wise_current_ui_mode = UI_MODE.SHOW_RESULTS;

	const search_keyword = Object.keys(response)[0];
	const results = response[search_keyword];

	toolbar.innerHTML = 'Search completed in ' + (search_time / 1000).toFixed(1) + ' sec.'
	for (var i = 0; i < results.length; ++i) {
		const img = document.createElement('img');
		const img_link = results[i]['link'];
		const img_link_tok = img_link.split('/');
		const img_filename = img_link_tok[img_link_tok.length - 2];

		img.src = results[i]['thumbnail'];
		img.setAttribute('title', 'File: ' + img_filename + ' | Distance = ' + results[i]['distance'].toFixed(2));

		const a = document.createElement('a');
		a.setAttribute('href', 'https://commons.wikimedia.org/wiki/File:' + img_filename);
		a.setAttribute('target', '_blank');
		a.appendChild(img)

		imgrid_container.appendChild(a);
	}
	navinfo1.innerHTML = 'Showing search results from ' + wise_result_start_findex + ' to ' + wise_result_end_findex + '.';
	navinfo2.innerHTML = 'Go back to <span onclick="load_featured_image_grid()" class="text_button">Home</span>';
}

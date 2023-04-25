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

// UI elements
const search_file_input = document.getElementById("search_file_input");

const UI_MODE = {
	'BROWSE_IMAGES': 'browse_images',
	'SHOW_RESULTS': 'show_results'
}

// state
var wise_home_featured_images = [];
var wise_home_from_findex = 0;
var wise_home_to_findex = PAGE_IMG_COUNT;
var wise_result_start_findex = -1;
var wise_result_end_findex = -1;
var wise_current_ui_mode = UI_MODE.BROWSE_IMAGES;
var wise_total_page_count = -1;
let search_query = {
	type: undefined,
	query: undefined,
};

//
// Home Page
//
const TOOLBAR_INFO = 'Try searching, for example, using <span class="text_button" onclick="search_for(\'bees feeding on flower\')">bees feeding on flower</span> or <span class="text_button" onclick="search_for(\'car on empty street in snow\')">car on empty street in snow</span> or <span class="text_button" onclick="search_for(\'horse in river\')">horse in river</span>.'
function init_home_page() {
	load_project_info();
	load_featured_images();
	document.getElementById('toolbar').innerHTML = TOOLBAR_INFO;
}

function load_project_info() {
	fetch("info", {
		method: 'GET'
	})
		.then((response) => response.json())
		.then((project_info) => {
			wise_data['info'] = project_info;

			// initialise the toolbar with info
			const img_count_percent = ((wise_data['info']['num_images'] / 80000000) * 100).toFixed(0);
			document.getElementById('toolbar').innerHTML = 'Here, you can search nearly ' + img_count_percent + '% of the 80 million images in Wikimedia Commons.' + TOOLBAR_INFO;
		});
}

function load_featured_images() {
	wise_current_ui_mode = UI_MODE.BROWSE_IMAGES;

	fetch("featured_images.json", {
		method: 'GET'
	})
		.then((response) => response.json())
		.then((featured_images_json) => {
			wise_home_featured_images = featured_images_json
				.map(value => ({ value, sort: Math.random() }))
				.sort((a, b) => a.sort - b.sort)
				.map(({ value }) => value); // Shuffle images
		
			wise_data['num_featured_images'] = wise_home_featured_images.length;
			show_featured_images(wise_home_from_findex, wise_home_to_findex);
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
		wise_home_from_findex = wise_home_to_findex;
		wise_home_to_findex = wise_home_to_findex + PAGE_IMG_COUNT;
		if (wise_home_to_findex > wise_data['num_featured_images']) {
			wise_home_from_findex = 0;
			wise_home_to_findex = PAGE_IMG_COUNT;
		}
		show_featured_images(wise_home_from_findex, wise_home_to_findex)
	} else {
		if (wise_result_start_findex < MAX_SEARCH_RESULT && wise_result_end_findex < MAX_SEARCH_RESULT) {
			wise_result_start_findex = wise_result_end_findex;
			wise_result_end_findex = wise_result_end_findex + SEARCH_RESULT_COUNT;
			if (wise_result_end_findex >= MAX_SEARCH_RESULT) {
				wise_result_end_findex = MAX_SEARCH_RESULT;
			}
			submit_search_query();
		} else {
			toolbar.innerHTML = 'No more search results are available';
		}
	}
}

function show_prev_page() {
	if (wise_current_ui_mode === UI_MODE.BROWSE_IMAGES) {
		wise_home_to_findex = wise_home_from_findex;
		wise_home_from_findex = wise_home_from_findex - PAGE_IMG_COUNT;
		if (wise_home_from_findex < 0) {
			wise_home_from_findex = wise_data['num_featured_images'] - PAGE_IMG_COUNT;
			wise_home_to_findex = wise_data['num_featured_images'];
		}
		show_featured_images(wise_home_from_findex, wise_home_to_findex)
	} else {
		if (wise_result_start_findex > 0) {
			wise_result_end_findex = wise_result_start_findex;
			wise_result_start_findex = wise_result_start_findex - SEARCH_RESULT_COUNT;
			if (wise_result_start_findex < 0) {
				wise_result_start_findex = 0;
			}
			submit_search_query();
		} else {
			toolbar.innerHTML = 'Already in the first page of the search results!';
		}
	}
}

function show_featured_images(from_findex, to_findex) {
	navinfo1.innerHTML = 'Showing featured images';
	imgrid_container.innerHTML = '';
	for (var i = from_findex; i < to_findex; ++i) {
		const img_link = wise_home_featured_images[i]['original_download_url']
		const img_link_tok = img_link.split('/');
		const img_filename = img_link_tok[img_link_tok.length - 2];

		const img = document.createElement('img');
		img.src = img_link;
		const a = document.createElement('a');
		a.setAttribute('href', 'https://commons.wikimedia.org/wiki/File:' + img_filename);
		a.setAttribute('target', '_blank');
		a.appendChild(img);

		imgrid_container.appendChild(a);
	}
	navinfo2.innerHTML = (from_findex + 1) + ' to ' + to_findex;
}

//
// Search query and results
//
function search_for(query) {
	document.getElementById('search_keyword').value = query;
	handle_search_submit();
}

function handle_search_submit() {
	search_query = {
		type: 'NATURAL_LANGUAGE',
		query: document.getElementById('search_keyword').value
	}
	submit_search_query({is_initial_query: true});
}

function submit_search_query({is_initial_query = false} = {}) {
	if (is_initial_query) {
		wise_result_start_findex = 0;
		wise_result_end_findex = SEARCH_RESULT_COUNT;
	}

	let searchMessage = '';
	if (is_initial_query) searchMessage += 'Searching ';
	else searchMessage += 'Continuing search ';

	if (search_query.type === 'NATURAL_LANGUAGE') {
		searchMessage += 'for <strong>' + search_query.query + '</strong> in ' + wise_data['info']['num_images'].toLocaleString('en', { useGrouping: true }) + ' Wikimedia images <div class="spinner"></div>';
	} else {
		searchMessage += 'for visually similar images in ' + wise_data['info']['num_images'].toLocaleString('en', { useGrouping: true }) + ' Wikimedia images <div class="spinner"></div>';
	}
	toolbar.innerHTML = searchMessage;
	
	const time0 = performance.now();
	send_search_request(wise_result_start_findex, wise_result_end_findex).then((search_result) => {
		const time1 = performance.now();
		const search_time = time1 - time0;
		show_search_result(search_result, search_time);
	}).catch((err) => {
		if (err instanceof DOMException) {
		} else {
			alert('An error has occurred. See the console for more details');
			console.error(err);
		}
	});
}

async function send_search_request(start, end) {
	let res;
	if (search_query.type === 'NATURAL_LANGUAGE') {
		res = await fetch(`search?q=${search_query.query}&start=${start}&end=${end}`, {
			method: 'GET'
		});
	} else {
		let formData = new FormData();
		formData.append('q', search_query.query);
		res = await fetch(`search?start=${start}&end=${end}`, {
			method: "POST",
			body: formData
		});
	}

	if (!res.ok) {
		const content_type = res.headers.get("content-type");
		let message = `${res.status} (${res.statusText})`;
		if (content_type && content_type.includes("application/json")) {
			const { detail } = await res.json();
			if ("message" in detail) {
				message = `${message} - ${detail["message"]}`;
			} else if (Array.isArray(detail)) {
				let err_message = "";
				detail.forEach(({ loc, msg }) => {
					err_message += `${loc.join("->")} - ${msg},`;
				});
				message = `${message} - (${err_message})`;
			}
		}
		throw new Error(message);
	}
	return res.json();
}

function handle_upload_button_click() {
	search_file_input.click();
}
search_file_input.onchange = async () => {
	search_query = {
		type: 'IMAGE',
		query: search_file_input.files[0]
	}
	submit_search_query({is_initial_query: true});
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
		const img_filename_decoded = decodeURIComponent(img_filename); // Decode filename to show special characters / utf-8 characters

		img.src = results[i]['thumbnail'];
		img.setAttribute('title', 'File: ' + img_filename_decoded + ' | Distance = ' + results[i]['distance'].toFixed(2));

		const a = document.createElement('a');
		a.setAttribute('href', 'https://commons.wikimedia.org/wiki/File:' + img_filename);
		a.setAttribute('target', '_blank');
		a.appendChild(img)

		imgrid_container.appendChild(a);
	}
	navinfo1.innerHTML = 'Showing search results from ' + (wise_result_start_findex + 1) + ' to ' + wise_result_end_findex + '.';
	navinfo2.innerHTML = 'Go back to <span onclick="load_featured_image_grid()" class="text_button">Home</span>';
}

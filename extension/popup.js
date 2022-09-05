$(function(){

    
    $('#keywordsubmit').click(function(){
		
		var search_topic = $('#keyword').val();
		
				
		if (search_topic){
                chrome.runtime.sendMessage(
					{text: search_topic},
					function(response) {
						result = response.farewell;
                        intro = "Named entities, format: (word, sentence number, word number of that sentence)"

                        people = ""
                        for (let i = 0; i < result.People.length; i +=1){
                            people = people.concat(" (", result.People[i], ")")
                        }

                        if (people == ""){
                            people = "No people detected"
                        }

                        locations = ""
                        for (let i = 0; i < result.Locations.length; i +=1){
                            locations = locations.concat(" (", result.Locations[i], ")")
                        }
                        if (locations == ""){
                            locations = "No locations detected"
                        }

                        organizations = ""
                        for (let i = 0; i < result.Organizations.length; i +=1){
                            organizations = organizations.concat(" (", result.Organizations[i], ")")
                        }
                        if (organizations == ""){
                            organizations = "No organizations detected"
                        }

                        other = ""
                        for (let i = 0; i < result.Other.length; i +=1){
                            other = other.concat(" (", result.Other[i], ")")
                        }
                        if (other == ""){
                            other = "No miscellaneous entities detected"
                        }

						alert(
                            intro.concat(
                                "\n\n", 
                                "People: ", people, "\n\n", 
                                "Locations: ", locations, "\n\n",
                                "Organizations: ", organizations, "\n\n",
                                "Other: ", other
                            )
                        )
					});
		}
			
			
		$('#keyword').val('');
		
    });
});
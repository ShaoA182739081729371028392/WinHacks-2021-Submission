import {Component} from "react"
import {View, Text, Image, StyleSheet, Linking} from 'react-native'
import {Helmet} from 'react-helmet'
const PATH_TO_SAMPLE= 'https://github.com/ShaoA182739081729371028392/WinHacks-2021-Submission/tree/main/Sample%20Images';
class App extends Component{
  onClick(event){
    if (event !== null){
      Linking.openURL(PATH_TO_SAMPLE);
    }
  }
  fileUploaded(event){
    var file = event.target.files[0]
    var form_data = new FormData()
    form_data.append('file', file);
    var fr = new FileReader()
    fr.onload = (eve) =>{
      fetch("http://localhost:5000/process", {
        method: "POST",
        body: form_data
      }).then((response) => {
        response.json().then((e) =>{
          
          let ingredients = e['Ingredient']
          let food_type = e['Class']
          
          var img = document.getElementById('image')
          img.innerHTML =  `<img src = ${eve.target.result} height = 200 width = 200/>`
          var ing = document.getElementById('ingredients')
          ing.innerHTML = `Ingredients: ${ingredients}`
          var cls = document.getElementById('class')
          cls.innerHTML =  `Food: ${food_type}`
          
        })
       
      })
    }
    if (file){
      fr.readAsDataURL(file)
    }
    
  }
  render(){
    return (
      <View style = {stylesheet.view}>
        <Helmet>
          <title>Pyra: Cooking Bot!</title>
        </Helmet>
        <Text style = {stylesheet.title}>Pyra: Ingredient and Food Prediction</Text>
        <Text>&nbsp;</Text>
        <Text style = {stylesheet.header}>
          Predict the type of food and a set of it's ingredients only from it's image. Predict from a set of 1013 ingredients and 101 types of food using Deep Learning.
        </Text>
        <Text>&nbsp;</Text>
        <Text style = {stylesheet.text}>
          This model achieves 55% accuracy(Random Guess Achieves 0.9% Accuracy) on the type of food and 90% accuracy(Random Guess is ~0% Accuracy) on the set of ingredients in the image. This model was created in 24 hours during WinHacks 2021. At inference time, it grabs the top 15 ingredients, meaning that some ingredients may be cut off or extra ingredients may be appended if there are very many or few ingredients in the recipe. Created By: Andrew Shao.
        </Text>
        <Text>&nbsp;</Text>
        <Text style = {stylesheet.text}>
          Model composed of a Transfer-Learned ResNet200D with Heavy Augmentation and Squeeze Excite Blocks, trained End-to-End as a multi-label classification problem. Feel Free to download sample images to use
          <span>
            &nbsp;<Text onClick = {this.onClick} style = {stylesheet.link}>here.</Text>
          </span>
        </Text>
        <span>&nbsp;</span>
        <Text style = {stylesheet.text}>Upload Food Image Here(Accepted Formats: JPG, PNG): </Text>
        <Text style = {stylesheet.text}>Note: This may take up to 10 seconds depending on how fast your browser is.</Text>
        <input type = 'file' name = 'file' onChange = {this.fileUploaded} accept = ".jpg, .png, .jpeg"/>
        <span>&nbsp;</span>
        <Text nativeID = "image"></Text>
        <Text text = {stylesheet.text} nativeID = "ingredients"></Text>
        <Text text = {stylesheet.text} nativeID = 'class'></Text>
      </View>
    )
  }
}
const stylesheet = StyleSheet.create({
  view: {
    padding: 50
  },
  text: {
    color: 'black',
    fontFamily: "Tahoma",
    fontSize: 20,
  },
  header: {
    fontFamily: "Tahoma",
    fontSize: 24,
    fontWeight: 'bold',
    color: "black"
  },
  link: {
    fontFamily: "Tahoma", 
    fontSize: 20,
    fontStyle: 'italic',
    color: 'blue',
    textDecorationLine: 'underline'
  },
  title: {
    fontWeight: 'bold',
    fontSize: 32,
    fontFamily: "Tahoma",
    color: "orange"
  }
})
export default App;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';


Future<String> fetchTranslation(originalTxt, lang) async{
  Map<String, String> body = {
    'original_text': originalTxt,
    'langauge': lang
  };
  final response = await http.post(
    'http://172.17.0.1:5000/translate-text',
    body: body
  );
  if (response.statusCode == 200){
    return _extractData(response)['text'];
  }else{
    return "Failed to translate";
  }
}


dynamic _extractData(http.Response resp) => jsonDecode(resp.body);
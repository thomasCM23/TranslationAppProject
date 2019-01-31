import 'package:flutter/material.dart';
import 'globals.dart' as gl;

class History {
  final String orignalTxt;
  final String toLang;
  final String itemID;

  History(this.orignalTxt, this.toLang, this.itemID);
}

class HistoryItem extends StatelessWidget {
  final History history;
  final Function removeFunc;
  final Function reloadFunc;

  const HistoryItem(this.history, this.removeFunc, this.reloadFunc);

  Widget _buildTiles(History item) {
    return RaisedButton(
      onPressed: reloadHistoryItem,
      color: Colors.white,
      child: Container(
          padding: EdgeInsets.symmetric(vertical: 10.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.start,
            children: <Widget>[
              Container(
                padding: EdgeInsets.only(right: 10.0),
                child: CircleAvatar(
                  child: Text(
                    item.toLang[0].toUpperCase(),
                    style: TextStyle(
                        fontSize: gl.largeText
                    ),
                  ),
                  backgroundColor: Colors.white54,
                  radius: 16.0,
                ),
              ),
              Flexible(
                child: Container(
                    margin: EdgeInsets.only(right: 1.0),
                    child: Text(
                      item.orignalTxt,
                      style: TextStyle(fontSize: gl.normaltext),
                      overflow: TextOverflow.ellipsis,
                    )
                ),
                flex: 20,
                fit: FlexFit.tight,
              ),
              Spacer(),
              Container(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: <Widget>[
                    IconButton(icon: Icon(Icons.clear), onPressed: removeHistoryItem,)
                  ],
                ),

              )

            ],
          )),
    );
  }

  void removeHistoryItem(){
    this.removeFunc(this.history.itemID);
  }
  void reloadHistoryItem(){
    this.reloadFunc(this.history.itemID);
    removeHistoryItem();
  }

  @override
  Widget build(BuildContext context) {
    return _buildTiles(history);
  }
}

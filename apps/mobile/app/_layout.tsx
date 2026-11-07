import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { LogBox } from 'react-native';

LogBox.ignoreLogs(['Created TensorFlow Lite']);

export default function RootLayout(){return <><StatusBar style="light"/><Stack screenOptions={{headerShown:false,contentStyle:{backgroundColor:'#07110F'}}}/></>}

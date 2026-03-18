import React from 'react';
import { View, FlatList, Image, Dimensions, StyleSheet, Text, useWindowDimensions } from 'react-native';

interface MangaReaderProps {
  pages: string[]; // Array of image URLs for manga pages
}

export default function MangaReader({ pages }: MangaReaderProps) {

    const { width: screenWidth, height: screenHeight } = useWindowDimensions();

    // Fallback
    if (pages.length === 0) {
        return (
            <View style={[styles.pageContainer, styles.centered]}>
                <Text>No pages available</Text>
            </View>
        );
    }

    const renderPage = ({ item, index }: { item: string, index: number }) => (
        <View style={[styles.pageContainer, { width: screenWidth, height: screenHeight }]}>
            <Image
                source={{ uri: item }}
                style={styles.page}
                resizeMode="contain"
            />
            {/* <Text style={styles.pageNumber}>Page {pages.indexOf(item) + 1}/{pages.length}</Text> */}
        </View>
    )

    return (
        <FlatList
            data={pages}
            keyExtractor={(item, index) => index.toString()}
            renderItem={renderPage}
            initialNumToRender={3}
            maxRenderPerBatch={5}
            windowSize={100}
        />
    );
}

const styles = StyleSheet.create({
    pageContainer: {
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#000',
    },
    page: {
        width: '100%',
        height: '100%',
    },
    pageNumber: {
        position: 'absolute',
        bottom: 10,
        color: 'rgba(255, 255, 255, 0.5)',
        fontSize: 12,
    },
    centered: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
});
import React from "react";
import {
  ActivityIndicator,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { bookmarkCardShadow } from "./bookmarkGridConstants";

export type CreateReadingListModalProps = {
  visible: boolean;
  creating: boolean;
  newListName: string;
  onChangeName: (name: string) => void;
  onClose: () => void;
  onSubmit: () => void;
  submitEnabled: boolean;
};

export default function CreateReadingListModal({
  visible,
  creating,
  newListName,
  onChangeName,
  onClose,
  onSubmit,
  submitEnabled,
}: CreateReadingListModalProps) {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={() => {
        if (!creating) onClose();
      }}
    >
      <View style={styles.root}>
        <Pressable
          style={styles.backdrop}
          onPress={onClose}
          accessibilityRole="button"
          accessibilityLabel="Close dialog"
        />
        <View style={styles.card}>
          <Text style={styles.title}>New reading list</Text>
          <Text style={styles.hint}>Name your list</Text>
          <TextInput
            style={styles.input}
            placeholder="List name"
            placeholderTextColor="#9ca3af"
            value={newListName}
            onChangeText={onChangeName}
            editable={!creating}
            autoFocus
            onSubmitEditing={() => void onSubmit()}
            returnKeyType="done"
          />
          <View style={styles.actions}>
            <Pressable
              style={({ pressed }) => [
                styles.btnGhost,
                pressed && styles.btnGhostPressed,
              ]}
              onPress={onClose}
              disabled={creating}
            >
              <Text style={styles.btnGhostText}>Cancel</Text>
            </Pressable>
            <Pressable
              style={({ pressed }) => [
                styles.btnPrimary,
                !submitEnabled && styles.btnPrimaryDisabled,
                pressed &&
                  submitEnabled &&
                  !creating &&
                  styles.btnPrimaryPressed,
              ]}
              onPress={() => void onSubmit()}
              disabled={!submitEnabled || creating}
            >
              {creating ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.btnPrimaryText}>Create</Text>
              )}
            </Pressable>
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 24,
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(17, 24, 39, 0.45)",
  },
  card: {
    width: "100%",
    maxWidth: 400,
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...bookmarkCardShadow,
  },
  title: {
    fontSize: 18,
    fontWeight: "800",
    color: "#111827",
    marginBottom: 4,
    letterSpacing: -0.2,
  },
  hint: {
    fontSize: 13,
    color: "#6b7280",
    marginBottom: 12,
  },
  input: {
    height: 48,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 12,
    paddingHorizontal: 14,
    fontSize: 16,
    color: "#111827",
    backgroundColor: "#f9fafb",
    marginBottom: 16,
  },
  actions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 10,
  },
  btnGhost: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    backgroundColor: "#fff",
    minWidth: 88,
    alignItems: "center",
    justifyContent: "center",
  },
  btnGhostPressed: { opacity: 0.85 },
  btnGhostText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#374151",
  },
  btnPrimary: {
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 10,
    backgroundColor: "#111827",
    minWidth: 96,
    alignItems: "center",
    justifyContent: "center",
  },
  btnPrimaryPressed: { opacity: 0.88 },
  btnPrimaryDisabled: { opacity: 0.5 },
  btnPrimaryText: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "600",
  },
});

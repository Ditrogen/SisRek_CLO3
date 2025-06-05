import tkinter as tk
from tkinter import ttk, messagebox
import threading
from rec_predictor import get_recommendations, initialize_recommender

class RecommenderApp:
    # Nginisiasi layout dari aplikasinya
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Rekomendasi RecBole")
        self.root.geometry("600x400") # Set ukuran jendela awal

        self.create_widgets()
        self.initialize_model_async()

    # Ini buat input fieldnya
    def create_widgets(self):
        # Frame untuk input
        input_frame = ttk.LabelFrame(self.root, text="Urutan Item Pengguna")
        input_frame.pack(padx=10, pady=10, fill="x")

        self.input_label = ttk.Label(input_frame, text="Masukkan urutan (misalnya, 1,2,3; 4,5):")
        self.input_label.pack(padx=5, pady=5, anchor="w")

        # Input urutan interaksi
        self.input_text = tk.Text(input_frame, height=5, width=70)
        self.input_text.pack(padx=5, pady=5)
        self.input_text.insert(tk.END, "1,2,3; 4,5") # Isi dengan contoh

        # Input nilai K
        k_frame = ttk.Frame(self.root)
        k_frame.pack(padx=10, pady=5, fill="x")

        self.k_label = ttk.Label(k_frame, text="Rekomendasi Top K:")
        self.k_label.pack(side="left", padx=5, pady=5)

        self.k_entry = ttk.Entry(k_frame, width=5)
        self.k_entry.insert(0, "10") # Nilai topK default
        self.k_entry.pack(side="left", padx=5, pady=5)

        # Tombol
        self.recommend_button = ttk.Button(self.root, text="Dapatkan Rekomendasi", command=self.get_recs_threaded)
        self.recommend_button.pack(padx=10, pady=10)

        # Label Status
        self.status_label = ttk.Label(self.root, text="Menginisialisasi model...", foreground="blue")
        self.status_label.pack(padx=10, pady=5)

        # Frame untuk output
        output_frame = ttk.LabelFrame(self.root, text="Rekomendasi")
        output_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.output_text = tk.Text(output_frame, height=10, width=70, state="disabled")
        self.output_text.pack(padx=5, pady=5, fill="both", expand=True)

    def initialize_model_async(self):
        # Threading
        """Menginisialisasi model dalam thread terpisah untuk mencegah GUI membeku."""
        self.recommend_button.config(state="disabled")
        self.status_label.config(text="Menginisialisasi model...", foreground="blue")
        threading.Thread(target=self._initialize_model_task, daemon=True).start()

    def _initialize_model_task(self):
        try:
            initialize_recommender()
            self.root.after(0, self._on_model_initialized, True)
        except Exception as e:
            self.root.after(0, self._on_model_initialized, False, str(e))

    def _on_model_initialized(self, success, error_message=None):
        if success:
            self.status_label.config(text="Model siap!", foreground="green")
            self.recommend_button.config(state="normal")
        else:
            self.status_label.config(text=f"Inisialisasi model gagal: {error_message}", foreground="red")
            messagebox.showerror("Kesalahan Inisialisasi", f"Gagal menginisialisasi model rekomendasi: {error_message}")

    def get_recs_threaded(self):
        """Memulai proses rekomendasi dalam thread baru."""
        self.recommend_button.config(state="disabled")
        self.status_label.config(text="Menghasilkan rekomendasi...", foreground="orange")
        threading.Thread(target=self._get_recommendations_task, daemon=True).start()

    def _get_recommendations_task(self):
        input_str = self.input_text.get("1.0", tk.END).strip()
        try:
            k = int(self.k_entry.get())
            if k <= 0:
                raise ValueError("Top K harus bilangan bulat positif.")
        except ValueError as e:
            self.root.after(0, self._display_error, f"Nilai K tidak valid: {e}")
            return

        user_item_sequences = []
        if input_str:
            try:
                # Pisahkan dengan titik koma untuk beberapa pengguna, lalu dengan koma untuk item
                user_strs = input_str.split(';')
                for user_str in user_strs:
                    item_ids_str = user_str.strip().split(',')
                    current_user_seq = []
                    for item_id_str in item_ids_str:
                        if item_id_str.strip(): # Hindari string kosong dari koma di akhir
                            current_user_seq.append(int(item_id_str.strip()))
                    if current_user_seq: # Hanya tambahkan jika ada item untuk pengguna
                        user_item_sequences.append(current_user_seq)
            except ValueError:
                self.root.after(0, self._display_error, "Format input tidak valid. Harap gunakan bilangan bulat yang dipisahkan koma dan titik koma (misalnya, 1,2,3; 4,5).")
                return

        if not user_item_sequences:
            self.root.after(0, self._display_error, "Tidak ada urutan item yang valid dimasukkan.")
            return

        try:
            recommendations = get_recommendations(user_item_sequences, top_k=k)
            self.root.after(0, self._display_results, recommendations)
        except Exception as e:
            self.root.after(0, self._display_error, f"Terjadi kesalahan selama rekomendasi: {e}")

    def _display_results(self, recommendations):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        if recommendations:
            for i, recs in enumerate(recommendations):
                self.output_text.insert(tk.END, f"Rekomendasi Pengguna {i+1} (Top {len(recs)}): {' ; '.join(recs)}\n")
        else:
            self.output_text.insert(tk.END, "Tidak ada rekomendasi ditemukan.\n")
        self.output_text.config(state="disabled")
        self.status_label.config(text="Rekomendasi siap!", foreground="green")
        self.recommend_button.config(state="normal")

    def _display_error(self, message):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Error: {message}\n")
        self.output_text.config(state="disabled")
        self.status_label.config(text=f"Error: {message}", foreground="red")
        self.recommend_button.config(state="normal")
        messagebox.showerror("Error", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderApp(root)
    root.mainloop()
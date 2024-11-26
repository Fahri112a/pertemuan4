def aStarAlgo(start_node, stop_node):
    open_set = set([start_node])  # Ubah menjadi list untuk inisialisasi set
    closed_set = set()

    g = {}  # Menyimpan jarak dari node awal
    parents = {}  # Parents berisi peta adjacency dari semua node

    # Jarak node awal dari dirinya sendiri adalah nol
    g[start_node] = 0

    # Node awal adalah root, jadi tidak memiliki parent node
    # Jadi start_node diatur sebagai parent dirinya sendiri
    parents[start_node] = start_node

    while len(open_set) > 0:
        n = None

        # Menemukan node dengan f(n) terendah
        for v in open_set:
            if n is None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v

        if n is None:
            print('Path does not exist!')
            return None

        # Jika node saat ini adalah tujuan, kita mulai merekonstruksi jalur dari tujuan ke awal
        if n == stop_node:
            path = []

            while parents[n] != n:
                path.append(n)
                n = parents[n]

            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path

        # Jika node tidak ada di graf atau tetangganya adalah None
        if n not in Graph_nodes or Graph_nodes[n] is None:
            print('Node {} tidak memiliki tetangga atau tidak ada dalam graf'.format(n))
            continue

        # Untuk setiap tetangga dari node n
        for (m, weight) in get_neighbors(n):
            # Jika node m tidak ada di open_set atau closed_set
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight

            # Jika node m sudah ada, bandingkan jarak dari start ke m
            else:
                if g[m] > g[n] + weight:
                    # Perbarui g(m)
                    g[m] = g[n] + weight
                    # Ubah parent m menjadi n
                    parents[m] = n

                    # Jika m ada di closed_set, pindahkan ke open_set
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

        # Hapus n dari open_set dan tambahkan ke closed_set karena semua tetangganya sudah diperiksa
        open_set.remove(n)
        closed_set.add(n)

    print('Path does not exist!')
    return None

# Fungsi untuk mengembalikan tetangga dan jarak dari node yang diberikan
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None

# Fungsi untuk mengembalikan jarak heuristik dari semua node
def heuristic(n):
    H_dist = {
        'A': 11,
        'B': 6,
        'C': 99,
        'D': 1,
        'E': 7,
        'G': 0,
    }

    return H_dist[n]

# Definisikan graf Anda di sini
Graph_nodes = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1), ('G', 9)],
    'C': None,
    'E': [('D', 6)],
    'D': [('G', 1)],
}

# Jalankan algoritma A*
aStarAlgo('A', 'G')
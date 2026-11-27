import { useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { changedLines, region } from '../../sim/bfsvisit';
import { functionSource } from '../../sim/toolchain';
import { BFS_TIERS } from './BfsGraph';
import { LevelExplorer, type ExplorerRequest } from './LevelExplorer';
import { VisitStepper } from './VisitStepper';
import { WavefrontScene } from './WavefrontScene';

// Section 18.3 of this course: the book's §18.3 to §18.5 (pp. 431-442). What the book says
// comes from notes/ch18.md; vendor facts from notes/today/ch18-facts.md (A6, A9, A12, B4, C4,
// C5, D1 and "Already settled"). Models: src/sim/bfs.ts (the explorer, the scene) and
// src/sim/bfsvisit.ts (the stepper). Listings: labs/ch18/snippets/s183_variants.cu.

export const questionsS183: Record<string, Question> = {
  'ch18/18.3/q-who-loops': {
    kind: 'choice',
    id: 'ch18/18.3/q-who-loops',
    prompt: (
      <Translated
        vi={
          <>
            Đồ thị của sách, gốc 0, kernel pull, lần gọi gán nhãn mức 2. Luồng nào quét toàn bộ danh sách cạnh vào của nó
            mà không gán nhãn gì, và vì sao không luồng nào khác như vậy?
          </>
        }
      >
        <>
          The book's graph, root 0, pull kernel, the call that labels level 2. Which thread scans its whole list of
          incoming edges and labels nothing, and why does no other?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Luồng của đỉnh 8: các đỉnh kề vào 3, 4 và 6 của nó chưa được thăm; mọi đỉnh chưa thăm khác đều có cha ở mức 1.">
            Vertex 8's: its in-neighbours 3, 4 and 6 are unvisited; every other unvisited vertex has a level-1 parent.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Luồng của đỉnh 0: đỉnh kề vào duy nhất của nó, 7, không ở mức 1, nên phép quét không tìm thấy cha.">
            Vertex 0's: its one in-neighbour, 7, is not at level 1, so its scan finds no parent.
          </Translated>
        ),
        why: (
          <Translated vi="Đỉnh 0 là gốc, ở mức 0. Luồng của nó không qua được kiểm tra chưa thăm ở dòng 05 và không bao giờ lặp.">
            Vertex 0 is the root, at level 0. Its thread fails the unvisited check on line 05 and never loops.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Luồng của đỉnh 4: các đỉnh kề vào của nó là 1 và 3, và nó phải kiểm tra cả hai trước khi gán nhãn.">
            Vertex 4's: its in-neighbours are 1 and 3, and it must check both before it labels.
          </Translated>
        ),
        why: (
          <Translated vi="Đỉnh kề vào đầu tiên của nó theo thứ tự CSC là 1, ở mức 1. Nó tự gán nhãn và thoát vòng lặp sau một cạnh.">
            Its first in-neighbour in CSC order is 1, at level 1. It labels itself and breaks after one.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không luồng nào: mọi luồng có lặp đều tìm thấy cha ở mức 1, vì mọi đỉnh của đồ thị này đều tới được từ 0.">
            None: every thread that loops finds a level-1 parent, since every vertex of this graph is reachable from 0.
          </Translated>
        ),
        why: (
          <Translated vi="Một luồng lặp mỗi khi đỉnh của nó chưa được thăm, dù đã có cha hay chưa. Các đỉnh kề vào của đỉnh 8 đều ở mức 2.">
            A thread loops whenever its vertex is unvisited, whether or not a parent exists yet. Vertex 8's in-neighbours are all at level 2.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong Hình 18.9, các luồng trên 3 đến 7 đều tìm thấy một cha ở mức 1 (1 hoặc 2) và dừng. Đỉnh 8 cách frontier hai
            mức, nên luồng của nó đọc cả ba cạnh vào một cách vô ích. Đó là sự lãng phí của pull ở đầu quá trình tìm kiếm: nó
            lặp cho mọi đỉnh chưa thăm, dù gần frontier hay không.
          </p>
        }
      >
        <p>
          In Fig. 18.9 the threads on 3 to 7 each find a parent at level 1 (1 or 2) and stop. Vertex 8 is two levels away
          from the frontier, so its thread reads all three in-edges for nothing. That is pull's waste early in a search:
          it loops for every unvisited vertex, near the frontier or not.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.3, pp. 434–435, Fig. 18.9',
  },
  'ch18/18.3/q-push-pull': {
    kind: 'choice',
    id: 'ch18/18.3/q-push-pull',
    prompt: (
      <Translated
        vi={
          <>
            Một đồ thị mạng xã hội ở mức 3 trên 6. Mức vừa rồi chứa 40% số đỉnh và 10% vẫn chưa được thăm. Push hay pull cho
            mức này?
          </>
        }
      >
        <>
          A social graph at level 3 of 6. The last level holds 40% of the vertices and 10% are still unvisited. Push or
          pull for this level?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Pull: chỉ 10% lặp, và với rất nhiều cha ở mức vừa rồi, phần lớn thoát ngay ở cạnh vào đầu tiên.">
            Pull: only the 10% loop, and with so many parents in the last level most break at their first in-edge.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Push: mức vừa rồi là nơi công việc mới bắt đầu, và 40% số đỉnh cho nó rất nhiều luồng.">
            Push: the last level is where the new work starts, and 40% of the vertices give it plenty of threads.
          </Translated>
        ),
        why: (
          <Translated vi="Push sẽ duyệt các danh sách cạnh ra dài của 40% đồ thị, phần lớn chỉ để thấy các đỉnh kề đã có nhãn.">
            Push would walk the long out-lists of 40% of the graph, mostly to find neighbours already labelled.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cái nào cũng được: trên cả quá trình tìm kiếm, cả hai đều đọc mỗi cạnh một lần, nên lựa chọn cho một mức không quan trọng.">
            Either: over a whole search both read every edge once, so the choice for one level does not matter.
          </Translated>
        ),
        why: (
          <Translated vi="Push đọc mỗi cạnh tổng cộng một lần; pull đọc lại các cạnh hết mức này đến mức khác. Xét từng mức, chi phí rất khác nhau.">
            Push reads each edge once in total; pull re-reads edges level after level. Per level, the costs are very different.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Push lặp cho các đỉnh của mức vừa rồi và duyệt mọi cạnh của chúng. Pull lặp cho các đỉnh chưa thăm và dừng ở cha
            đầu tiên. Về cuối một quá trình tìm kiếm trên đồ thị có bậc cao và không đều, các đỉnh chưa thăm còn ít còn cha
            của chúng thì nhiều, và đó là lúc BFS tối ưu theo hướng chuyển kernel.
          </p>
        }
      >
        <p>
          Push loops for the last level's vertices and walks every edge they have. Pull loops for the unvisited and stops
          at the first parent. Late in a search on a graph with high, uneven degrees, the unvisited are few and their
          parents plentiful, which is the case direction-optimised BFS switches for.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.3, pp. 435–436',
  },
  'ch18/18.3/q-ex2': {
    kind: 'choice',
    id: 'ch18/18.3/q-ex2',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 2: bạn viết vòng lặp host của một BFS tối ưu theo hướng. Cái gì phải có trên device, và ở mỗi mức host
            cần gì để chọn kernel?
          </>
        }
      >
        <>
          Exercise 2: you write the host loop of a direction-optimised BFS. What must be on the device, and what does the
          host need at each level to choose the kernel?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="CSR và CSC (một cặp nếu đồ thị đối xứng), và ở mỗi mức một số đo như số cạnh ra khỏi frontier so với số cạnh ra khỏi các đỉnh chưa thăm.">
            CSR and CSC (one pair if symmetric), and per level a measure such as edges out of the frontier against edges out of unvisited vertices.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ CSR, vì pull có thể quét các hàng để tìm cha; và số thứ tự mức, để chuyển ở một mức cố định.">
            CSR only, since pull can scan the rows for its parents; and the level number, to switch at a fixed level.
          </Translated>
        ),
        why: (
          <Translated vi="Pull đọc các cạnh vào, mà CSR chỉ cho được bằng cách tìm trong mọi hàng. Và mức phù hợp phụ thuộc vào đồ thị, điều mà một con số cố định bỏ qua.">
            Pull reads incoming edges, which CSR gives only by searching every row. And the right level depends on the graph, which a fixed number ignores.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ COO, vì một danh sách cặp (src, dst) phục vụ được cả push lẫn pull, và kích thước frontier mà host vốn đã đọc.">
            COO only, since one list of (src, dst) pairs serves push and pull alike, and the frontier size the host already reads.
          </Translated>
        ),
        why: (
          <Translated vi="COO phục vụ kernel lấy cạnh làm trung tâm. Push cần các cạnh ra của một đỉnh dưới dạng một đoạn, pull cần các cạnh vào; COO không cho cái nào nếu không quét.">
            COO serves the edge-centric kernel. Push needs a vertex's out-edges as a slice, pull its in-edges; COO gives neither without a scan.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách chỉ nói rằng điểm chuyển đến sớm hơn với đồ thị có bậc cao và không đều. Quy tắc của cuGraph, theo Beamer
            2012, so sánh số cạnh ra khỏi frontier với số cạnh ra khỏi các đỉnh chưa thăm: m_f và m_u trong bảng của trình
            khám phá. Host phải lấy những con số đó về ở mỗi mức, thêm một lần đọc ngoài cờ hoặc kích thước frontier.
          </p>
        }
      >
        <p>
          The book says only that the switch comes earlier for graphs with high, uneven degrees. cuGraph's rule, after
          Beamer 2012, compares edges out of the frontier with edges out of unvisited vertices: m_f and m_u in the
          explorer's table. The host has to bring those numbers back each level, one more read on top of the flag or
          frontier size.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.3, p. 436; Exercise 2, p. 450; cuGraph: notes/today/ch18-facts.md D1',
    vi: { source: 'ch. 18 §18.3, p. 436; Bài tập 2, p. 450; cuGraph: notes/today/ch18-facts.md D1' },
  },
  'ch18/18.3/q-hub-checks': {
    kind: 'numeric',
    id: 'ch18/18.3/q-hub-checks',
    prompt: (
      <Translated
        vi={
          <>
            Một đỉnh có 1,000 cạnh ra không bao giờ thuộc mức vừa rồi trong một quá trình tìm kiếm mà vòng lặp host thực hiện
            10 lần gọi kernel, lần cuối không tìm thấy gì. Tổng cộng, các luồng của kernel lấy cạnh làm trung tâm đọc mức của
            nó bao nhiêu lần?
          </>
        }
      >
        <>
          A vertex with 1,000 out-edges is never in the last level during a search whose host loop makes 10 kernel
          calls, the last one finding nothing. How many times, in total, do edge-centric threads read its level?
        </>
      </Translated>
    ),
    answer: 10000,
    unit: 'reads',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi cạnh trong 1,000 cạnh của nó có một luồng ở mọi lần gọi, và mỗi luồng đọc mức của đỉnh nguồn để biết rằng nó
            không có việc gì làm: 1,000 cạnh × 10 lần gọi. Một luồng push đọc nó một lần mỗi lần gọi, tổng cộng 10, và bỏ qua
            cả danh sách. Mười lần gọi là một quá trình tìm kiếm 9 mức sau gốc, vì lần gọi cuối không tìm thấy gì.
          </p>
        }
      >
        <p>
          Each of its 1,000 edges has a thread in every call, and each thread reads the source's level to find out it has
          nothing to do: 1,000 edges × 10 calls. A push thread reads it once per call, 10 in all, and skips the whole
          list. Ten calls is a search of 9 levels after the root, since the last call finds nothing.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.4, p. 438',
    vi: { unit: 'lần đọc' },
  },
  'ch18/18.3/q-edge-gain': {
    kind: 'choice',
    id: 'ch18/18.3/q-edge-gain',
    prompt: (
      <Translated vi={<>Khi nào BFS lấy cạnh làm trung tâm thắng push lấy đỉnh làm trung tâm?</>}>
        <>When does edge-centric BFS beat vertex-centric push?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trên một đồ thị nhỏ có bậc lệch: nhiều luồng hơn để lấp đầy thiết bị, và không có danh sách dài nào kéo dài một luồng.">
            On a small graph with skewed degrees: more threads to fill the device, and no long list stretching one thread.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trên một mạng lưới đường bộ lớn, vì nó đọc mỗi cạnh đúng một lần trên cả quá trình tìm kiếm, như push dùng frontier.">
            On a large road network, because it reads every edge exactly once over the whole search, like push with frontiers.
          </Translated>
        ),
        why: (
          <Translated vi="Nó đọc mọi cạnh ở mọi mức, và một mạng lưới đường bộ có nhiều mức.">
            It reads every edge at every level, and a road network has many levels.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bất cứ khi nào thiếu bộ nhớ, vì COO, không có mảng con trỏ theo đỉnh, là định dạng nhỏ nhất trong ba.">
            Whenever memory is short, because COO, with no pointer array per vertex, is the smallest of the three formats.
          </Translated>
        ),
        why: (
          <Translated vi="COO lưu hai số cho mỗi cạnh; CSR và CSC lưu một số cho mỗi cạnh cộng một số cho mỗi đỉnh.">
            COO stores two numbers per edge; CSR and CSC store one per edge plus one per vertex.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi luồng một cạnh cho ra m luồng thay vì n, và mọi luồng làm cùng một lượng việc không đổi, nên một đỉnh trung
            tâm (hub) không thể giữ chân warp của nó. Cái giá là phải kiểm tra cả m cạnh ở mọi mức và các mảng COO lớn hơn.
          </p>
        }
      >
        <p>
          One thread per edge exposes m threads instead of n, and every thread does the same constant work, so a hub
          cannot hold up its warp. The price is checking all m edges at every level and the larger COO arrays.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.4, pp. 437–438',
  },
  'ch18/18.3/q-chain': {
    kind: 'choice',
    id: 'ch18/18.3/q-chain',
    prompt: (
      <Translated
        vi={
          <>
            Một chuỗi n đỉnh, tìm kiếm từ một đầu, nên d = n − 1. Push lấy đỉnh làm trung tâm khởi chạy tổng cộng bao nhiêu
            luồng, so với phiên bản dùng frontier?
          </>
        }
      >
        <>
          A chain of n vertices, searched from one end, so d = n − 1. How many threads does vertex-centric push launch in
          total, against the frontier version?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Khoảng n² so với khoảng n: O(d·n) so với O(n + m).">About n² against about n: O(d·n) against O(n + m).</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Mỗi bên khoảng n: cả hai đều thăm mỗi đỉnh một lần.">About n each: both visit every vertex once.</Translated>,
        why: (
          <Translated vi="Push khởi chạy một luồng cho mọi đỉnh ở mọi mức, dù đã thăm hay chưa.">
            Push launches a thread for every vertex at every level, visited or not.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Khoảng n log n so với n.">About n log n against n.</Translated>,
        why: (
          <Translated vi="Ở đây không có gì giảm một nửa: chuỗi có n − 1 mức, mỗi mức một đỉnh.">
            Nothing halves here: the chain has n − 1 levels, one vertex each.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            n luồng mỗi lần gọi trong n lần gọi là n². Phiên bản dùng frontier khởi chạy một luồng cho đỉnh duy nhất của mỗi
            mức. Các đồ thị giống mạng đường bộ, với nhiều mức ít đỉnh, là nơi BFS lấy đỉnh làm trung tâm kém hiệu quả về công
            việc nhất. Hãy thử chuỗi trong trình khám phá.
          </p>
        }
      >
        <p>
          n threads per call for n calls is n². The frontier version launches one thread for the single vertex of each
          level. Road-like graphs, with many levels of few vertices, are where vertex-centric BFS is at its least
          work-efficient. Try the chain in the explorer.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.5, pp. 438–439',
  },
  'ch18/18.3/q-why-cas': {
    kind: 'choice',
    id: 'ch18/18.3/q-why-cas',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao trên thực tế Hình 18.6 vẫn ổn với một phép kiểm tra rồi gán nhãn không đồng bộ, trong khi Hình 18.12 phải
            dùng compare-and-swap?
          </>
        }
      >
        <>
          Why does Fig. 18.6 get away, in practice, with an unsynchronised check-then-label, while Fig. 18.12 must use
          compare-and-swap?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trong Hình 18.6, một bản trùng chỉ ghi lại cùng một mức. Trong Hình 18.12, mỗi luồng thắng còn nối thêm vào frontier, nên một bản trùng bị mở rộng hai lần.">
            In Fig. 18.6 a duplicate rewrites the same level. In Fig. 18.12 each winner also appends, so a duplicate is expanded twice.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Hình 18.12 khởi chạy ít luồng hơn nhiều ở mỗi mức, nên khả năng hai luồng cùng tới một đỉnh kề cao hơn.">
            Fig. 18.12 launches far fewer threads per level, so two of them reaching one neighbour together is likelier.
          </Translated>
        ),
        why: (
          <Translated vi="Ít luồng hơn làm va chạm hiếm hơn, không phải dễ xảy ra hơn. Khác biệt nằm ở hậu quả của một va chạm.">
            Fewer threads make collisions rarer, not likelier. The difference is what a collision does.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hình 18.6 không có race theo mô hình bộ nhớ, vì mọi lần ghi tranh chấp đều ghi cùng một giá trị; Hình 18.12 không có bảo đảm đó.">
            Fig. 18.6 is race-free under the memory model, because every racing write stores the same value; Fig. 18.12 has no such guarantee.
          </Translated>
        ),
        why: (
          <Translated vi="Hình 18.6 cũng là một race condition (tranh chấp dữ liệu), không được định nghĩa theo mô hình bộ nhớ. Nó chỉ chạy được trên thực tế.">
            Fig. 18.6 is a data race too, undefined by the memory model. It merely works in practice.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Không có frontier thì một bản trùng là lũy đẳng. Có frontier thì mỗi lần kiểm tra rồi gán nhãn còn chiếm một ô, và
            chiếm hai lần thì không lũy đẳng. Về hình thức Hình 18.6 vẫn là một race condition, vì vậy bản đã sửa dùng atomic
            relaxed.
          </p>
        }
      >
        <p>
          Without frontiers a duplicate is idempotent. With frontiers each check-then-label also claims a slot, and
          claiming twice is not idempotent. Fig. 18.6 is still formally a data race, which is why the corrected listing
          uses relaxed atomics.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.5, pp. 440–441',
  },
  'ch18/18.3/q-stale': {
    kind: 'choice',
    id: 'ch18/18.3/q-stale',
    prompt: (
      <Translated
        vi={
          <>
            Đoạn mã khai báo <code>unsigned int e = UNVISITED;</code> một lần, phía trên vòng lặp cạnh. Luồng trên đỉnh 2 gặp
            đỉnh kề 3, đã bị một luồng khác giành ở mức này, rồi đến 4, chưa thăm. Điều gì xảy ra?
          </>
        }
      >
        <>
          The code declares <code>unsigned int e = UNVISITED;</code> once, above the edge loop. The thread on vertex 2
          meets neighbour 3, already won by another thread this level, and then 4, unvisited. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phép swap thất bại trên 3 ghi mức của 3 vào e. Sau đó phép swap trên 4 so sánh UNVISITED với mức đó và thất bại: 4 bị bỏ sót.">
            The failed swap on 3 writes 3's level into e. The swap on 4 then compares UNVISITED with that level and fails: 4 is missed.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả hai đều chạy đúng: một compare-and-swap thất bại đặt lại e về UNVISITED, sẵn sàng cho đỉnh kề kế tiếp.">
            Both work: a failed compare-and-swap resets e to UNVISITED, ready for the next neighbour.
          </Translated>
        ),
        why: (
          <Translated vi="Nó làm điều ngược lại: khi thất bại, nó nạp giá trị nó tìm thấy vào e.">
            It does the opposite: on failure it loads the value it found into e.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="3 bị gán nhãn lại: phép swap trên 3 so sánh với UNVISITED của lời khai báo, nên nó ghi đè luồng thắng.">
            3 is relabelled: the swap on 3 compares with the UNVISITED of the declaration, so it overwrites the winner.
          </Translated>
        ),
        why: (
          <Translated vi="Phép swap trên 3 thất bại, vì level[3] không còn giữ UNVISITED, và chính lần thất bại đó làm thay đổi e. Hãy chạy từng bước trong trình chạy từng bước để thấy.">
            The swap on 3 fails, because level[3] no longer holds UNVISITED, and that failure is what changes e. Step it in the stepper to see.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sau một compare-and-swap thất bại, biến expected giữ giá trị hiện tại (ở đây là mức 2). Dùng lại nó nghĩa là so
            sánh UNVISITED của 4 với 2: không bằng nhau, nên phép swap lại thất bại và nạp UNVISITED trở lại vào e, nhưng đã
            quá muộn. Đỉnh 4 được gán nhãn ở một mức sau đó, hoặc không bao giờ: trong đồ thị này chỉ có 2 trỏ tới 4. Hình
            18.14 tránh điều này bằng cách khai báo biến của nó mới ở mỗi lần gọi.
          </p>
        }
      >
        <p>
          After a failed compare-and-swap the expected variable holds the current value (here, level 2). Reusing it
          compares 4's UNVISITED with 2: not equal, so the swap fails again and loads UNVISITED back into e, too late.
          Vertex 4 is labelled at a later level, or never: in this graph only 2 points to 4. Fig. 18.14 avoids this by
          declaring its variable fresh on every call.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.5, p. 441; C++ semantics: notes/today/ch18-facts.md A6',
    vi: { source: 'ch. 18 §18.5, p. 441; ngữ nghĩa C++: notes/today/ch18-facts.md A6' },
  },
  'ch18/18.3/q-weak-loop': {
    kind: 'choice',
    id: 'ch18/18.3/q-weak-loop',
    prompt: (
      <Translated
        vi={
          <>
            Metal chỉ có compare-and-swap yếu, có thể thất bại ngay cả khi các giá trị bằng nhau. Vòng lặp nào thử lại đúng và
            chỉ những lần thất bại giả?
          </>
        }
      >
        <>
          Metal has only a weak compare-and-swap, which may fail even when the values are equal. Which loop retries
          exactly the spurious failures?
        </>
      </Translated>
    ),
    options: [
      {
        label: 'while (!cas(e) && e == UNVISITED) {}',
        correct: true,
      },
      {
        label: 'while (!cas(e)) {}', // i18n-ignore: code
        why: (
          <Translated vi="Sau một lần thất bại thật, e giữ mức của luồng thắng, nên lần thử lại so sánh mức đó với chính nó, thành công, và nối thêm một bản trùng, hoặc ghi đè một mức cũ hơn bằng mức hiện tại.">
            After a real failure e holds the winner's level, so the retry compares that level with itself, succeeds, and appends a duplicate, or overwrites an older level with the current one.
          </Translated>
        ),
      },
      {
        label: 'while (!cas(e)) e = UNVISITED;',
        why: (
          <Translated vi="Một khi luồng khác đã thắng, level[v] không bao giờ là UNVISITED nữa, nên vòng lặp này không bao giờ kết thúc.">
            Once another thread has won, level[v] is never UNVISITED again, so this never ends.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lần thất bại giả để <code>e</code> ở UNVISITED; một lần thất bại thật nạp mức của luồng thắng vào nó. Kiểm tra{' '}
            <code>e</code> giúp phân biệt hai trường hợp: thử lại với trường hợp đầu, bỏ cuộc với trường hợp sau, và chỉ nối
            vào frontier nếu phép swap thành công. Hãy thử cả ba trong trình chạy từng bước với một lần thất bại giả cho phép.
          </p>
        }
      >
        <p>
          A spurious failure leaves <code>e</code> at UNVISITED; a real one loads the winner's level into it. Testing{' '}
          <code>e</code> tells the two apart: retry on the first, give up on the second, and append only if the swap
          succeeded. Try all three in the stepper with one spurious failure allowed.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch18-facts.md A6, B4, C4 (MSL spec §6.16.4.4)',
    vi: { source: 'notes/today/ch18-facts.md A6, B4, C4 (đặc tả MSL §6.16.4.4)' },
  },
};

// ---------------------------------------------------------------- the five listings
const FIGS = [
  { id: 'fig18_6', get label() { return tr('Fig. 18.6 push', 'Hình 18.6 push'); } },
  { id: 'fig18_8', get label() { return tr('Fig. 18.8 pull', 'Hình 18.8 pull'); } },
  { id: 'fig18_10', get label() { return tr('Fig. 18.10 edge', 'Hình 18.10 edge'); } },
  { id: 'fig18_12', get label() { return tr('Fig. 18.12 frontier', 'Hình 18.12 frontier'); } },
  { id: 'fig18_14', get label() { return tr('Fig. 18.14 visit', 'Hình 18.14 visit'); } },
];
const WHAT_CHANGED: Record<string, string> = {
  get fig18_6() {
    return tr(
      'Lines 05 and 09 to 11: level[] and the flag through relaxed atomic loads and stores.',
      'Dòng 05 và 09 đến 11: level[] và cờ được đọc, ghi qua các phép load và store atomic relaxed.',
    );
  },
  get fig18_8() {
    return tr('Lines 05 and 09 to 11: the same, for the pull kernel.', 'Dòng 05 và 09 đến 11: tương tự, cho kernel pull.');
  },
  get fig18_10() {
    return tr('Lines 06 and 08 to 10: the same, for the edge-centric kernel.', 'Dòng 06 và 08 đến 10: tương tự, cho kernel lấy cạnh làm trung tâm.');
  },
  get fig18_12() {
    return tr(
      'No line changes: the frontier kernel touches level[] only through Fig. 18.14.',
      'Không dòng nào thay đổi: kernel frontier chỉ chạm vào level[] thông qua Hình 18.14.',
    );
  },
  get fig18_14() {
    return tr(
      'A HIP branch: the legacy atomicCAS, which is strong on HIP. The CUDA lines are the book\'s.',
      'Một nhánh HIP: atomicCAS kiểu cũ (legacy), vốn là dạng mạnh trên HIP. Các dòng CUDA là của sách.',
    );
  },
};

function Listings() {
  const src = useLabFile('ch18/snippets/s183_variants.cu');
  const [fixed, setFixed] = useState(false);
  const [which, setWhich] = useState(0);
  if (!src) return null;
  const tabs = FIGS.map((f) => {
    const printed = region(src, f.id + '_printed') ?? '// region not found';
    const corrected = region(src, f.id + '_fixed') ?? '// region not found';
    return {
      label: f.label,
      code: fixed ? corrected : printed,
      tiers: BFS_TIERS,
      highlight: fixed ? changedLines(printed, corrected) : [],
      status: tr(
        'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware',
        'emulator-checked và front-end-checked; unverified trên phần cứng CUDA và HIP',
      ),
    };
  });
  return (
    <div className="wide" style={{ maxWidth: '58rem' }}>
      <div className="row ui" style={{ marginBottom: '-0.4rem' }}>
        <span className="seg" role="group" aria-label={tr('Listing version', 'Phiên bản mã')}>
          <button aria-pressed={!fixed} onClick={() => setFixed(false)}><Translated vi="như bản in">as printed</Translated></button>
          <button aria-pressed={fixed} onClick={() => setFixed(true)}><Translated vi="đã sửa">corrected</Translated></button>
        </span>
        <span className="seg" role="group" aria-label={tr('Figure', 'Hình')}>
          {FIGS.map((f, i) => (
            <button key={f.id} aria-pressed={which === i} onClick={() => setWhich(i)}>{f.label.split(' ')[1]}</button>
          ))}
        </span>
      </div>
      <Code
        key={which}
        tabs={[tabs[which]]}
        caption={
          <Translated
            vi={
              <>
                {FIGS[which].label}, {fixed ? 'đã sửa' : 'như bản in'}, từ <code>labs/ch18/snippets/s183_variants.cu</code>.{' '}
                {fixed ? <>Được đánh dấu: những gì đã thay đổi. {WHAT_CHANGED[FIGS[which].id]}</> : <>Văn bản và số dòng của sách. File thêm các struct đồ thị và <code>UNVISITED</code> mà sách không bao giờ định nghĩa.</>}
              </>
            }
          >
            <>
              {FIGS[which].label}, {fixed ? 'corrected' : 'as printed'}, from <code>labs/ch18/snippets/s183_variants.cu</code>.{' '}
              {fixed ? <>Marked: what changed. {WHAT_CHANGED[FIGS[which].id]}</> : <>The book's text and line numbers. The file adds the graph structs and the <code>UNVISITED</code> the book never defines.</>}
            </>
          </Translated>
        }
      />
    </div>
  );
}

/**
 * The Metal tab's status, from this machine's reports: lab 18.1's "provided frontier" test runs
 * ref_bfs_frontier, which calls ref_visit_vertex, in every variant.
 */
function metalVisitStatus(reports: Reports): string {
  const rs = (reports.metal?.results ?? []).filter((r) => r.lab === 'ch18/lab1_frontier' && r.built);
  const tests = rs.flatMap((r) => r.tests.filter((t) => t.case.startsWith('provided frontier')));
  if (!tests.length) return tr('unverified: no run of lab 18.1 on this machine', 'unverified: chưa có lần chạy lab 18.1 nào trên máy này');
  return tests.every((t) => t.pass)
    ? tr(
        "verified on Metal (lab 18.1's provided frontier test, which calls it, passes on this machine)",
        'verified on Metal (bài kiểm tra provided frontier của lab 18.1, vốn gọi hàm này, đạt trên máy này)',
      )
    : tr('unverified: failed on this machine', 'unverified: thất bại trên máy này');
}

/** Code inside the backends table: allowed to wrap, so no column is squeezed by one long name. */
const TC = ({ children }: { children: ReactNode }) => <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;

/** Fig. 18.14 as the course writes it for CUDA and HIP, and the Metal form lab 18.1 provides. */
function VisitOnBackends() {
  const cu = useLabFile('ch18/snippets/s183_variants.cu');
  const metal = useLabFile('ch18/lab1_frontier/provided/ref_kernels.metal');
  const { reports } = useReports();
  if (!cu || !metal) return null;
  const fig = region(cu, 'fig18_14_fixed') ?? '// region not found';
  const fn = functionSource(metal, 'ref_visit_vertex');
  return (
    <Code
      tabs={[
        { label: tr('CUDA and HIP', 'CUDA và HIP'), code: fig, tiers: BFS_TIERS, highlight: changedLines(region(cu, 'fig18_14_printed') ?? '', fig), status: tr('emulator-checked and front-end-checked; unverified on CUDA and HIP hardware', 'emulator-checked và front-end-checked; unverified trên phần cứng CUDA và HIP') },
        { label: 'Metal', tiers: BFS_TIERS, code: fn ? fn.text : '// ref_visit_vertex not found in labs/ch18/lab1_frontier/provided/ref_kernels.metal', status: metalVisitStatus(reports) },
      ]}
      caption={
        <Translated
          vi={
            <>
              Tab CUDA là Hình 18.14 đã sửa, lấy từ <code>labs/ch18/snippets/s183_variants.cu</code>, có đánh dấu nhánh HIP.
              Tab Metal là hàm thăm đỉnh mà lab 18.1 cung cấp, lấy từ{' '}
              <code>labs/ch18/lab1_frontier/provided/ref_kernels.metal</code>: vòng lặp thử lại, viết dưới dạng return sớm.
            </>
          }
        >
          <>
            The CUDA tab is the corrected Fig. 18.14 from <code>labs/ch18/snippets/s183_variants.cu</code>, its HIP branch
            marked. The Metal tab is the visit function lab 18.1 provides, from{' '}
            <code>labs/ch18/lab1_frontier/provided/ref_kernels.metal</code>: the retry loop, written as an early return.
          </>
        </Translated>
      }
    />
  );
}

export function S183() {
  const [request, setRequest] = useState<ExplorerRequest | null>(null);
  const show = (variant: 'pull' | 'edge', step: number) => setRequest({ preset: 1, root: 0, variant, step, key: Date.now() });
  return (
    <>
      <Translated
        vi={
          <p>
            Trang này chạy một mức BFS theo bốn cách trên bốn đồ thị nhỏ, đếm công việc mà mỗi cách làm, và kết thúc bằng
            bước mà mọi kernel về sau đều dựa vào: làm cho một đỉnh vào frontier kế tiếp đúng một lần, bằng compare-and-swap,
            trên CUDA, HIP và Metal.
          </p>
        }
      >
        <p>
          This page runs one level of BFS four ways on four small graphs, counts the work each way does, and ends with
          the step every later kernel depends on: making a vertex join the next frontier exactly once, with
          compare-and-swap, on CUDA, HIP and Metal.
        </p>
      </Translated>

      <h2><Translated vi="Một lần khởi chạy cho mỗi mức">One launch per level</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi kernel trên trang này gán nhãn một mức mỗi lần khởi chạy. Gốc bắt đầu ở mức 0 và mọi đỉnh khác ở{' '}
            <code>UNVISITED</code>. Kernel cho mức L gán nhãn các đỉnh kề chưa thăm của các đỉnh mức L − 1, và host khởi chạy
            lại nó cho L + 1 cho đến khi một lần khởi chạy không gán nhãn gì. Ranh giới kernel đang làm việc thật: nó là thứ
            duy nhất bảo đảm mọi đỉnh của mức L − 1 đã được gán nhãn trước khi bất kỳ luồng nào tìm mức L. Mục 18.7 thay nó
            bằng một rào chắn (barrier) bên trong một lần khởi chạy.
          </p>
        }
      >
        <p>
          Every kernel on this page labels one level per launch. The root starts at level 0 and every other vertex at{' '}
          <code>UNVISITED</code>. The kernel for level L labels the unvisited neighbours of the level L − 1 vertices, and
          the host launches it again for L + 1 until a launch labels nothing. The kernel boundary is doing real work: it
          is the only thing that guarantees every vertex of level L − 1 is labelled before any thread looks for level L.
          Section 18.7 replaces it with a barrier inside one launch.
        </p>
      </Translated>
      <Source section="18.3" pages="432" />
      <Translated
        vi={
          <p>
            Năm kernel của mục này ở bên dưới, như bản in và như khóa học này sửa lại. Hãy đọc Hình 18.6 trước; các hình khác
            sẽ lần lượt được nhắc tới.
          </p>
        }
      >
        <p>
          The section's five kernels are below, as printed and as this course corrects them. Read Fig. 18.6 first; the
          others come up in turn.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Sách không bao giờ định nghĩa các struct đồ thị hay <code>UNVISITED</code>; p. 433 gợi ý{' '}
              <code>cuda::std::numeric_limits&lt;unsigned&gt;</code>, vốn chỉ gọi tên trait mà thiếu{' '}
              <code>max()</code>. File dùng <code>UINT_MAX</code>.
            </p>
          }
        >
          <p>
            The book never defines its graph structs or <code>UNVISITED</code>; p. 433 suggests{' '}
            <code>cuda::std::numeric_limits&lt;unsigned&gt;</code>, which names the trait without its{' '}
            <code>max()</code>. The file uses <code>UINT_MAX</code>.
          </p>
        </Translated>
      </Aside>
      <Listings />

      <h2><Translated vi="Push: các luồng trên mức vừa rồi">Push: threads on the last level</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 18.6 cho mỗi đỉnh một luồng. Một luồng có đỉnh ở mức L − 1 duyệt các cạnh ra của nó trong CSR (dòng 06 đến 08)
            và gán nhãn mọi đỉnh kề vẫn còn <code>UNVISITED</code> (dòng 09 và 10). Nó cũng bật{' '}
            <code>newVertexVisited</code>, cờ mà host đọc để quyết định có khởi chạy lại hay không. Các luồng đứng trên cha và
            tìm con, nên sách gọi đây là top-down, hay push. Với gốc 0, trong lần gọi gán nhãn mức 2, chỉ các luồng trên 1 và
            2 vượt qua dòng 05: đó là Hình 18.7.
          </p>
        }
      >
        <p>
          Fig. 18.6 gives every vertex a thread. A thread whose vertex is at level L − 1 walks its out-edges in CSR (lines
          06 to 08) and labels every neighbour still <code>UNVISITED</code> (lines 09 and 10). It also sets{' '}
          <code>newVertexVisited</code>, the flag the host reads to decide whether to launch again. The threads sit on
          parents and look for children, so the book calls this top-down, or push. With root 0, in the call that labels
          level 2 only the threads on 1 and 2 get past line 05: that is Fig. 18.7.
        </p>
      </Translated>
      <Source section="18.3" pages="432–433" figs={tr('Figs. 18.6, 18.7', 'Hình 18.6, 18.7')} />
      <Translated
        vi={
          <p>
            Hai luồng có thể tới cùng một đỉnh kề chưa thăm, cả hai thấy <code>UNVISITED</code>, và cả hai ghi cùng một mức.
            Sách gọi đây là vô hại: lần ghi là lũy đẳng, nên trên thực tế không có gì sai. Sách nói thêm rằng đoạn mã vi phạm
            mô hình bộ nhớ C++ và không được bảo đảm sẽ chạy đúng, và một lập trình viên cẩn thận sẽ dùng atomic. Bản đã sửa
            làm đúng như vậy: các phép load và store atomic relaxed trên <code>level[]</code> và cờ, không đòi hỏi thứ tự nào
            và loại bỏ race.
          </p>
        }
      >
        <p>
          Two threads can reach the same unvisited neighbour, both see <code>UNVISITED</code>, and both write the same
          level. The book calls this benign: the write is idempotent, so in practice nothing goes wrong. It adds that the
          code violates the C++ memory model and is not guaranteed to work, and that a careful programmer uses atomics.
          The corrected listing does: relaxed atomic loads and stores of <code>level[]</code> and the flag, which ask for
          no ordering and remove the race.
        </p>
      </Translated>
      <Source section="18.3" pages="433–434" />
      <Today
        checked={tr(
          '2026-11-20 (notes/today/ch18-facts.md, “Already settled”: ch10 A1, ch14 A9)',
          '2026-11-20 (notes/today/ch18-facts.md, mục “Already settled”: ch10 A1, ch14 A9)',
        )}
      >
        <Translated
          vi={
            <p>
              Mô hình bộ nhớ của CUDA coi một race condition là hành vi không xác định, dù các lần ghi tranh chấp có ghi cùng
              một giá trị hay không. HIP không ghi nhận quy tắc riêng nào.
            </p>
          }
        >
          <p>
            CUDA's memory model makes a data race undefined behaviour whether or not the racing writes store the same
            value. HIP documents no rule of its own.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Pull: các luồng trên đỉnh chưa thăm">Pull: threads on the unvisited</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 18.8 đảo ngược lại. Một luồng có đỉnh vẫn còn <code>UNVISITED</code> quét các cạnh vào của nó trong CSC và
            tìm một đỉnh kề ở mức L − 1. Một là đủ: nó gán nhãn cho chính đỉnh của mình và thoát vòng lặp (dòng 12). Mỗi luồng
            chỉ ghi mức của chính nó, nên cờ là lần ghi chung duy nhất. Các luồng đứng trên con và tìm cha: bottom-up, hay
            pull.
          </p>
        }
      >
        <p>
          Fig. 18.8 turns it round. A thread whose vertex is still <code>UNVISITED</code> scans its in-edges in CSC and
          looks for a neighbour at level L − 1. One is enough: it labels its own vertex and breaks (line 12). Each thread
          writes only its own level, so the flag is the only shared write. The threads sit on children looking for a
          parent: bottom-up, or pull.
        </p>
      </Translated>
      <Source section="18.3" pages="434–435" figs={tr('Figs. 18.8, 18.9', 'Hình 18.8, 18.9')} />
      <Ask q={questionsS183['ch18/18.3/q-who-loops']} />
      <Translated
        vi={
          <p>
            Cách nào tốt hơn phụ thuộc vào mức, vì hai lý do. Push duyệt cả một danh sách cạnh ra; pull có thể dừng ở cha đầu
            tiên, điều ít quan trọng trên mạng đường bộ nhưng rất quan trọng trên đồ thị mạng xã hội có danh sách dài và không
            đều. Và push lặp cho các đỉnh của mức vừa rồi, còn pull lặp cho mọi đỉnh chưa được thăm. Ở đầu quá trình tìm kiếm,
            mức vừa rồi nhỏ và phần lớn đỉnh chưa thăm, nên push thắng. Về sau, mức vừa rồi lớn và còn ít đỉnh, mỗi đỉnh nhiều
            khả năng tìm thấy cha ngay, nên pull thắng.
          </p>
        }
      >
        <p>
          Which is better depends on the level, for two reasons. Push walks a whole out-list; pull can stop at its first
          parent, which matters little on a road network and a lot on a social graph with long, uneven lists. And push
          loops for the vertices of the last level, pull for every vertex not yet visited. Early in a search the last
          level is small and most vertices are unvisited, so push wins. Late, the last level is large and few vertices
          are left, each likely to find a parent at once, so pull wins.
        </p>
      </Translated>
      <Source section="18.3" pages="435–436" />
      <Translated
        vi={
          <p>
            BFS tối ưu theo hướng (direction-optimised) dùng push trước rồi chuyển sang pull. Nó cần cạnh ra cho các mức push
            và cạnh vào cho các mức pull: cả CSR lẫn CSC, trừ khi đồ thị đối xứng (mọi cạnh đều có cạnh ngược), khi đó một cặp
            mảng phục vụ cả hai. Về thời điểm chuyển, sách chỉ nói rằng đồ thị có bậc cao và không đều chuyển sớm hơn.
          </p>
        }
      >
        <p>
          Direction-optimised BFS pushes first and switches to pull. It needs out-edges for the push levels and in-edges
          for the pull levels: CSR and CSC both, unless the graph is symmetric (every edge has its reverse), when one pair
          of arrays serves both. On when to switch, the book says only that graphs with high, uneven degrees switch
          earlier.
        </p>
      </Translated>
      <Source section="18.3" pages="436" />
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md D1)', '2026-11-20 (notes/today/ch18-facts.md, mục D1)')}>
        <Translated
          vi={
            <p>
              BFS của cuGraph là loại tối ưu theo hướng, theo Beamer 2012, và đòi hỏi đồ thị đối xứng. Nó chuyển sang bottom-up
              khi số cạnh ra khỏi frontier (m_f), nhân với một hệ số đã tinh chỉnh, nhiều hơn số cạnh ra khỏi các đỉnh chưa thăm
              (m_u). Trình khám phá bên dưới hiển thị cả hai con số ở mọi lần gọi.
            </p>
          }
        >
          <p>
            cuGraph's BFS is direction-optimising, after Beamer 2012, and requires a symmetric graph. It goes bottom-up
            when the edges out of the frontier (m_f), scaled by a tuned factor, outnumber the edges out of unvisited
            vertices (m_u). The explorer below shows both counts at every call.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS183['ch18/18.3/q-push-pull']} />
      <Ask q={questionsS183['ch18/18.3/q-ex2']} label={tr('Exercise 2, p. 450', 'Bài tập 2, p. 450')} />

      <h2><Translated vi="Lấy cạnh làm trung tâm: mỗi cạnh một luồng">Edge-centric: a thread per edge</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 18.10 cho mỗi cạnh một luồng. Luồng đọc đỉnh nguồn của cạnh từ COO; nếu đỉnh nguồn ở mức L − 1 và đỉnh đích
            vẫn còn <code>UNVISITED</code>, nó gán nhãn đỉnh đích. Không có vòng lặp, nên không luồng nào bị kéo dài bởi một
            danh sách dài: mọi luồng làm cùng một lượng việc, đó là ý tưởng ánh xạ lại luồng với công việc của chương 6. Cũng
            có nhiều luồng hơn, giúp ích cho một đồ thị quá nhỏ để lấp đầy thiết bị với mỗi đỉnh một luồng. Chi phí: mọi cạnh
            bị kiểm tra ở mọi mức, và COO lưu hai số cho mỗi cạnh.
          </p>
        }
      >
        <p>
          Fig. 18.10 gives every edge a thread. It reads its edge's source from COO; if the source is at level L − 1 and
          the destination is still <code>UNVISITED</code>, it labels the destination. There is no loop, so no thread is
          stretched by a long list: every thread does the same work, which is chapter 6's idea of re-mapping threads to
          work. There are also more threads, which helps a graph too small to fill the device one thread per vertex. The
          costs: every edge is checked at every level, and COO stores two numbers per edge.
        </p>
      </Translated>
      <Source section="18.4" pages="436–438" figs={tr('Figs. 18.10, 18.11', 'Hình 18.10, 18.11')} />
      <Translated
        vi={
          <p>
            Ví dụ của sách cho chi phí thứ nhất là một đỉnh có nhiều cạnh nhưng không liên quan ở một mức nào đó. Kernel lấy
            cạnh làm trung tâm khởi chạy một luồng cho mỗi cạnh, và mỗi luồng đều phát hiện cùng một điều; push phát hiện một
            lần và bỏ qua cả danh sách.
          </p>
        }
      >
        <p>
          The book's example of the first cost is a vertex with many edges that is irrelevant at some level. Edge-centric
          launches a thread per edge, and each one finds out the same thing; push finds out once and skips the list.
        </p>
      </Translated>
      <Source section="18.4" pages="438" />
      <Ask q={questionsS183['ch18/18.3/q-hub-checks']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS183['ch18/18.3/q-edge-gain']} />
      <Translated
        vi={
          <p>
            Các kernel này trông giống các kernel SpMV của chương 17 là có lý do: một mức BFS có thể viết thành một phép nhân
            ma trận thưa với vectơ, cách nhìn đằng sau API GraphBLAS. Sách dành cho nó một đoạn văn; khóa học này dừng ở đó.
          </p>
        }
      >
        <p>
          These kernels look like chapter 17's SpMV kernels for a reason: a BFS level can be written as a sparse
          matrix–vector product, the view behind the GraphBLAS API. The book gives it a paragraph; this course leaves it
          there.
        </p>
      </Translated>
      <Source section="18.4" pages="438" />
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md A12)', '2026-11-20 (notes/today/ch18-facts.md, mục A12)')}>
        <Translated
          vi={<p>GraphBLAS C API 2.1.0; bản cài đặt tham chiếu của nó chạy trên CPU, thư mục CUDA của nó là bản nháp.</p>}
        >
          <p>GraphBLAS C API 2.1.0; its reference implementation runs on the CPU, its CUDA folder a draft.</p>
        </Translated>
      </Today>

      <h2><Translated vi="Chạy chúng cạnh nhau">Run them side by side</Translated></h2>
      <Translated
        vi={
          <p>
            Bài tập 1 hỏi, với mỗi lần gọi trên đồ thị 8 đỉnh của nó, mỗi kernel khởi chạy bao nhiêu luồng, bao nhiêu luồng
            lặp và bao nhiêu luồng gán nhãn. Hãy dự đoán hai trong số đó; mỗi lần xem đáp án sẽ mở trình khám phá ở đúng lần
            gọi đó.
          </p>
        }
      >
        <p>
          Exercise 1 asks, for every call on its 8-vertex graph, how many threads each kernel launches, how many loop and
          how many label. Predict two of those counts; each reveal opens the explorer on that call.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Đáp án là của chúng tôi: sách không in đáp án nào, và hướng các cạnh là cách chúng tôi đọc hình vẽ ở p. 450. Các
              unit test của mô hình đối chiếu chúng với ghi chú.
            </p>
          }
        >
          <p>
            The answers are ours: the book prints none, and the edge directions are our reading of the drawing on p.
            450. The model's unit tests check them against the notes.
          </p>
        </Translated>
      </Aside>
      <Predict
        id="ch18/18.3/p-ex1c-pull"
        prompt={
          <Translated
            vi={<>Đồ thị Bài tập 1, gốc 0, kernel pull, lần gọi thứ hai (gán nhãn mức 2). Bao nhiêu luồng lặp qua các cạnh vào của chúng?</>}
          >
            <>Exercise 1 graph, root 0, pull kernel, second call (it labels level 2). How many threads loop over their incoming edges?</>
          </Translated>
        }
        unit={tr('threads', 'luồng')}
      >
        {(v) => (
          <>
            <Translated
              vi={
                <p>
                  {v === 5 ? 'Đúng: ' : v === 3 ? 'Đó là số luồng gán nhãn. Đáp án là ' : 'Đáp án là '}
                  5: các đỉnh chưa thăm 1, 3, 4, 6 và 7. Trong số đó, 3 luồng tìm thấy cha ở mức 1 và gán nhãn cho đỉnh của mình
                  (1, 3 và 7); 4 và 6 chưa có cha nào.
                </p>
              }
            >
              <p>
                {v === 5 ? 'Right: ' : v === 3 ? 'That counts the threads that label. The answer is ' : 'It is '}
                5: the unvisited vertices 1, 3, 4, 6 and 7. Of those, 3 find a parent at level 1 and label their vertex
                (1, 3 and 7); 4 and 6 have none yet.
              </p>
            </Translated>
            <button onClick={() => show('pull', 2)}><Translated vi="Xem trong trình khám phá">Show it in the explorer</Translated></button>
          </>
        )}
      </Predict>
      <Predict
        id="ch18/18.3/p-ex1c-edge"
        prompt={
          <Translated vi={<>Cùng đồ thị, kernel lấy cạnh làm trung tâm, lần gọi thứ ba. Bao nhiêu luồng thấy đỉnh nguồn của cạnh mình ở mức 2?</>}>
            <>Same graph, edge-centric kernel, third call. How many threads find their edge's source at level 2?</>
          </Translated>
        }
        unit={tr('threads', 'luồng')}
      >
        {(v) => (
          <>
            <Translated
              vi={
                <p>
                  {v === 8 ? 'Đúng: ' : v === 4 ? 'Đó là số luồng có thể gán nhãn. Đáp án là ' : 'Đáp án là '}
                  8: các cạnh ra khỏi 1, 3 và 7. Bốn trong số đó có thể gán nhãn: 1 → 4, 7 → 4, 3 → 6 và 7 → 6, hai luồng ghi cho
                  mỗi đỉnh 4 và 6, lại là bản trùng vô hại.
                </p>
              }
            >
              <p>
                {v === 8 ? 'Right: ' : v === 4 ? 'That counts the threads that may label. The answer is ' : 'It is '}
                8: the edges out of 1, 3 and 7. Four of them may label: 1 → 4, 7 → 4, 3 → 6 and 7 → 6, two writers for each
                of 4 and 6, the benign duplicate again.
              </p>
            </Translated>
            <button onClick={() => show('edge', 3)}><Translated vi="Xem trong trình khám phá">Show it in the explorer</Translated></button>
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Những thứ nên thử. Đồ thị của sách, gốc 0, lần gọi 2, push rồi pull: Hình 18.7 và 18.9. Chuỗi, push so với
            frontier: theo dõi tổng số luồng đã khởi chạy. Hình sao bắt đầu từ một lá, push ở lần gọi 2: luồng của đỉnh trung
            tâm lặp một mình trong khi warp của nó chờ, còn kernel lấy cạnh làm trung tâm trải cùng các cạnh đó ra mỗi luồng
            một cạnh. Tối ưu theo hướng với thanh trượt điểm chuyển, theo dõi m_f và m_u.
          </p>
        }
      >
        <p>
          Things to try. The book's graph, root 0, call 2, push and then pull: Figs. 18.7 and 18.9. The chain, push
          against frontier: watch the launched totals. The star from a leaf, push at call 2: the hub's thread loops alone
          while its warp waits, and edge-centric spreads the same edges one per thread. Direction-optimised with the
          switch slider, watching m_f and m_u.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một bước là một lần gọi kernel; không có gì được đo thời gian. Khối và warp rộng 4 luồng, để cả một lần gọi vừa
              trên màn hình. Mọi luồng của một lần gọi đều thấy các mức như trước lần gọi, nên khi hai luồng có thể ghi cùng
              một đỉnh, số nhãn được ghi là một khoảng. Kernel frontier nối vào theo một thứ tự có thể (các luồng theo thứ tự
              chỉ số). Hình 18.7 đến 18.13 in mức cuối cùng của mỗi đỉnh ngay cả ở bước từ mức 1 sang mức 2; trình khám phá hiển
              thị các mức tại lần gọi hiện tại.
            </p>
          }
        >
          <p>
            A step is one kernel call; nothing is timed. Blocks and warps are 4 threads wide, so a whole call fits on
            screen. Every thread of a call sees the levels as they were before the call, so where two threads may write the same vertex the labels written are a range. The frontier kernel
            appends in one possible order (threads in index order). Figs. 18.7 to 18.13 print each vertex's final level
            even at the level 1 to 2 step; the explorer shows levels as of the current call.
          </p>
        </Translated>
      </Simplification>
      <LevelExplorer request={request} />

      <h2><Translated vi="Hiệu quả công việc">Work efficiency</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi kernel cho đến giờ đều kiểm tra mọi đỉnh hoặc mọi cạnh ở mọi mức. Điều đó giữ cho chúng đơn giản và không cần
            đồng bộ hóa, nhưng làm phần lớn luồng của chúng vô dụng. Với một đồ thị có n đỉnh, m cạnh và d mức sau mức của gốc,
            sách đếm như sau:
          </p>
        }
      >
        <p>
          Every kernel so far checks every vertex or every edge at every level. That keeps them simple and free of
          synchronisation, and makes most of their threads useless. For a graph with n vertices, m edges and d levels
          after the root's, the book counts:
        </p>
      </Translated>
      <Source section="18.5" pages="438–439" />
      <table className="data">
        <thead>
          <tr>
            <th><Translated vi="Kernel">Kernel</Translated></th>
            <th><Translated vi="Công việc">Work</Translated></th>
            <th><Translated vi="Vì sao">Why</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="lý tưởng">ideal</Translated></td>
            <td>O(n + m)</td>
            <td><Translated vi="mỗi đỉnh và cạnh một lần">each vertex and edge once</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="push lấy đỉnh làm trung tâm">vertex-centric push</Translated></td>
            <td>O(d·n + m)</td>
            <td>
              <Translated vi="mọi đỉnh bị kiểm tra ở mỗi mức; mỗi cạnh được duyệt một lần, từ mức của đỉnh nguồn">
                every vertex checked each level; each edge walked once, from its source's level
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="pull lấy đỉnh làm trung tâm">vertex-centric pull</Translated></td>
            <td>O(d·n + d·m)</td>
            <td>
              <Translated vi="mọi đỉnh ở mỗi mức; một cạnh có thể bị đọc lại ở mọi mức cho đến khi đỉnh đích được gán nhãn">
                every vertex each level; an edge can be re-read every level until its destination is labelled
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="lấy cạnh làm trung tâm">edge-centric</Translated></td>
            <td>O(d·m)</td>
            <td><Translated vi="mọi cạnh ở mỗi mức">every edge each level</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="push dùng frontier">push with frontiers</Translated></td>
            <td>O(n + m)</td>
            <td><Translated vi="chỉ có luồng cho các đỉnh của mức vừa rồi">threads only for the last level's vertices</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Chạy bất kỳ kernel nào đến hết trong trình khám phá và nó sẽ nêu tổng của nó so với cận. Trên chuỗi, khác biệt
            giữa d·n và n + m là khác biệt giữa n² và n.
          </p>
        }
      >
        <p>
          Run any kernel to the end in the explorer and it states its total against the bound. On the chain, the
          difference between d·n and n + m is the difference between n² and n.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Lỗi nhỏ ở p. 439: "the threads whose vertices are not in the <i>current</i> level do not do anything" ý nói mức
              trước, như dòng 05 của Hình 18.6 cho thấy. Cận của pull ở đó cũng thiếu cân bằng dấu ngoặc.
            </p>
          }
        >
          <p>
            Slip on p. 439: "the threads whose vertices are not in the <i>current</i> level do not do anything" means the
            previous level, as line 05 of Fig. 18.6 shows. The pull bound there also has an unbalanced parenthesis.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS183['ch18/18.3/q-chain']} />

      <h2><Translated vi="Frontier">Frontiers</Translated></h2>
      <Translated
        vi={
          <p>
            Cách khắc phục là chỉ khởi chạy luồng cho các đỉnh được gán nhãn ở lần gọi trước. Trong lúc chạy, các luồng đó dựng
            danh sách các đỉnh chúng gán nhãn, tức frontier, và lần khởi chạy kế tiếp nhận mỗi phần tử một luồng. Hình 18.12
            nhận <code>prevFrontier</code> với <code>numPrevFrontier</code> phần tử, và <code>currFrontier</code> với một bộ
            đếm <code>numCurrFrontier</code> trong bộ nhớ toàn cục. Để nối thêm, một luồng thực hiện <code>fetch_add(1)</code>{' '}
            trên bộ đếm, phép này trả về giá trị cũ, tức ô của riêng nó, rồi lưu đỉnh vào đó (dòng 12 đến 16). Đó là phép lọc
            không ổn định của chương 12: các phần tử đến theo bất kỳ thứ tự nào mà các atomic xảy ra.
          </p>
        }
      >
        <p>
          The fix is to launch threads only for the vertices labelled in the last call. Those threads build, as they go,
          the list of vertices they label, the frontier, and the next launch gets one thread per entry. Fig. 18.12 takes{' '}
          <code>prevFrontier</code> with <code>numPrevFrontier</code> entries, and <code>currFrontier</code> with a
          counter <code>numCurrFrontier</code> in global memory. To append, a thread does <code>fetch_add(1)</code> on the
          counter, which returns the old value, its own slot, and stores the vertex there (lines 12 to 16). That is
          chapter 12's unstable filter: the entries arrive in whatever order the atomics happen.
        </p>
      </Translated>
      <Source section="18.5" pages="439–440" figs={tr('Figs. 18.12, 18.13', 'Hình 18.12, 18.13')} />
      <Translated
        vi={
          <p>
            Mốc dự án M5 dùng cùng hình dạng mã này để nén gọn các token sống sót sau một lần cắt top-p thành một danh sách dày
            đặc: chiếm một ô bằng một atomic, rồi lưu id của token.
          </p>
        }
      >
        <p>
          Milestone M5 uses the same code shape to compact the tokens that survive a top-p cut into a dense list: claim a
          slot with one atomic, then store the token id.
        </p>
      </Translated>
      <Beyond title={tr('Bridge (ours)', 'Cầu nối (của khóa học này)')}>
        <Translated
          vi={
            <p>
              Các kernel lấy mẫu không có trong sách; <a href={href('ch12', 'labs')}>các lab của chương 12</a> dựng danh sách
              ứng viên của M5.
            </p>
          }
        >
          <p>
            The sampling kernels are not in the book; <a href={href('ch12', 'labs')}>chapter 12's labs</a> build the M5
            candidate list.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Cờ không còn nữa: host dừng khi một lần khởi chạy để frontier trống. Một quá trình tìm kiếm d mức vẫn cần d + 1 lần
            khởi chạy, lần cuối không tìm thấy gì, và mỗi lần đều cần host đọc kích thước frontier về trước. Hoạt hình đặt hai
            kernel và các lần khởi chạy cạnh nhau trên một lưới đi dây.
          </p>
        }
      >
        <p>
          The flag is gone: the host stops when a launch leaves the frontier empty. A search of d levels still takes d + 1
          launches, the last one finding nothing, and each needs the host to read the frontier size back first. The
          animation puts the two kernels and the launches side by side on a wiring grid.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt hình đếm mỗi ô một luồng, không làm tròn lên thành khối đầy đủ, và không đưa ra thời gian nào. Lưới của nó
              là bố cục riêng của chúng tôi theo tinh thần của Hình 18.5, không phải hình của sách.
            </p>
          }
        >
          <p>
            The animation counts one thread per cell, without rounding up to whole blocks, and gives no times. Its grid is
            our own layout in the spirit of Fig. 18.5, not the book's figure.
          </p>
        </Translated>
      </Simplification>
      <WavefrontScene />

      <h2><Translated vi="Đúng một lần">Exactly once</Translated></h2>
      <Translated
        vi={
          <p>
            Không có frontier thì một nhãn trùng là vô hại. Có frontier thì không: hai luồng cùng thấy một đỉnh kề là{' '}
            <code>UNVISITED</code> sẽ cùng nối nó vào, và lần khởi chạy kế tiếp mở rộng nó hai lần. Vì vậy kiểm tra và gán
            nhãn phải là một bước không thể chia cắt. Compare-and-swap chính là bước đó: so sánh <code>level[v]</code> với{' '}
            <code>UNVISITED</code> và, chỉ khi chúng bằng nhau, ghi <code>currLevel</code>, đồng thời báo lại nó có ghi hay
            không. Hình 18.14 bọc nó trong <code>visitVertexAtomically</code>, và trong mọi luồng tranh nhau một đỉnh, đúng một
            luồng nhận true.
          </p>
        }
      >
        <p>
          Without frontiers a duplicate label was harmless. With them it is not: two threads that both see a neighbour as{' '}
          <code>UNVISITED</code> both append it, and the next launch expands it twice. So checking and labelling must be
          one indivisible step. Compare-and-swap is that step: compare <code>level[v]</code> with <code>UNVISITED</code>
          and, only if they are equal, write <code>currLevel</code>, reporting whether it did. Fig. 18.14 wraps it in{' '}
          <code>visitVertexAtomically</code>, and of all the threads that race for a vertex exactly one gets true.
        </p>
      </Translated>
      <Source section="18.5" pages="440–441" figs={tr('Fig. 18.14', 'Hình 18.14')} />
      <Translated
        vi={
          <p>
            Ba chi tiết. Giá trị expected được truyền theo tham chiếu, nên nó phải là một biến, không phải hằng{' '}
            <code>UNVISITED</code>. Một compare-and-swap thất bại ghi giá trị nó tìm thấy vào biến đó, nên một biến được dùng
            lại cho đỉnh kề kế tiếp không còn giữ <code>UNVISITED</code>; Hình 18.14 khai báo nó mới ở mỗi lần gọi, và điều đó
            đúng. Và thứ tự relaxed cho cả khi thành công lẫn khi thất bại là hợp lệ: không có gì khác cần được sắp thứ tự với
            vị trí duy nhất này.
          </p>
        }
      >
        <p>
          Three details. The expected value is passed by reference, so it must be a variable, not the constant{' '}
          <code>UNVISITED</code>. A failed compare-and-swap writes the value it found into that variable, so a variable
          reused for the next neighbour no longer holds <code>UNVISITED</code>; Fig. 18.14 declares it fresh on every
          call, which is right. And relaxed ordering on success and on failure is valid: nothing else needs ordering
          against this one location.
        </p>
      </Translated>
      <Source section="18.5" pages="441–442" />
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md A6)', '2026-11-20 (notes/today/ch18-facts.md, mục A6)')}>
        <Translated
          vi={
            <p>
              libcu++ khai báo tham số expected là một tham chiếu không const. Sau một lần thất bại, nó giữ giá trị đã tìm
              thấy, như trong C++ chuẩn. Dùng relaxed cho cả hai thứ tự là được phép.
            </p>
          }
        >
          <p>
            libcu++ declares the expected parameter as a non-const reference. After a failure it holds the value that was
            found, as in standard C++. Relaxed for both orders is allowed.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Sách dự kiến tranh chấp vừa phải trên compare-and-swap, vì chỉ các luồng có chung một đỉnh kề mới gặp nhau ở đó, và
            tranh chấp cao trên bộ đếm, nơi mọi luồng nối vào đều chạm tới. Bộ đếm đó là vấn đề của trang sau.
          </p>
        }
      >
        <p>
          The book expects moderate contention on the compare-and-swap, since only threads with a common neighbour meet
          there, and high contention on the counter, which every appending thread hits. That counter is the next page's
          problem.
        </p>
      </Translated>
      <Source section="18.5" pages="442" />
      <Aside>
        <Translated
          vi={
            <p>
              Lỗi nhỏ ở p. 441: "the vertex-centric push kernel without frontiers in Fig. 18.12" ý nói Hình 18.6, có dòng 09
              và 10 là phép kiểm tra và phép gán nhãn.
            </p>
          }
        >
          <p>
            Slip on p. 441: "the vertex-centric push kernel without frontiers in Fig. 18.12" means Fig. 18.6, whose lines
            09 and 10 are the check and the label.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS183['ch18/18.3/q-why-cas']} />
      <Translated
        vi={
          <p>
            Trình chạy từng bước đưa hai hoặc ba luồng của một mức tới một đỉnh kề chung và để bạn chọn ai chạy khi nào. Bắt
            đầu với phép kiểm tra thường và một lần nối: tìm một thứ tự đưa một đỉnh vào frontier hai lần. Sau đó chuyển sang
            compare-and-swap và bấm <i>Thử mọi thứ tự</i>. Rồi đưa <code>e</code> lên trên vòng lặp và chạy thứ tự xuôi trên
            tình huống cái bẫy.
          </p>
        }
      >
        <p>
          The stepper takes two or three threads of one level to a shared neighbour and lets you choose who moves when.
          Start with the plain check and an append: find an order that puts a vertex in the frontier twice. Then switch to
          compare-and-swap and press <i>Try every order</i>. Then hoist <code>e</code> above the loop and run the forward
          order on the trap scenario.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một bước là một câu lệnh, và nó không thể chia cắt: một load, một store, một compare-and-swap, hoặc một lần nối
              (một <code>fetch_add</code> và lần store vào ô của nó). Mọi load đều thấy store mới nhất; không có hiệu ứng thứ tự
              bộ nhớ nào. Một compare-and-swap yếu chỉ thất bại giả ở nơi bạn cho nó thất bại.
            </p>
          }
        >
          <p>
            One step is one statement, and it is indivisible: a load, a store, a compare-and-swap, or an append (a{' '}
            <code>fetch_add</code> and the store into its slot). Every load sees the latest store; there are no
            memory-order effects. A weak compare-and-swap fails spuriously only where you make it.
          </p>
        </Translated>
      </Simplification>
      <VisitStepper />
      <Ask q={questionsS183['ch18/18.3/q-stale']} label={tr('Bug hunt', 'Săn lỗi')} />

      <h2><Translated vi="Trên HIP và Metal">On HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Cùng một compare-and-swap được viết theo ba cách, và nó không mạnh như nhau ở cả ba. Một compare-and-swap mạnh chỉ
            thất bại khi các giá trị khác nhau. Một compare-and-swap yếu còn có thể thất bại khi chúng bằng nhau. Hình 18.14
            dựa vào dạng mạnh: một lần thất bại giả sẽ để đỉnh đó lại cho một mức sau, hoặc không mức nào.
          </p>
        }
      >
        <p>
          The same compare-and-swap is spelled three ways, and it is not equally strong in all three. A strong
          compare-and-swap fails only when the values differ. A weak one may also fail when they are equal. Fig. 18.14
          relies on the strong form: a spurious failure would leave the vertex for a later level, or for none.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Mọi điều ở đây về HIP và Metal đều lấy từ file dữ kiện và chưa được chạy trên phần cứng AMD
              hay Apple.
            </p>
          }
        >
          <p>
            The book uses CUDA only. Everything here about HIP and Metal is from the facts file and was not run on AMD or
            Apple hardware.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên CUDA, <code>cuda::atomic_ref::compare_exchange_strong</code> là dạng mạnh. Trên HIP cùng dòng đó vẫn biên dịch
            được, nhưng libhipcxx cài đặt nó ở phạm vi khối và thiết bị bằng built-in dạng yếu; <code>atomicCAS</code> kiểu cũ
            (legacy) là dạng mạnh và trả về giá trị cũ. Vì vậy các file <code>.cu</code> của khóa học dùng{' '}
            <code>atomicCAS</code> dưới <code>__HIPCC__</code>, như lab 12.1 dùng các atomic kiểu cũ. Metal chỉ có{' '}
            <code>atomic_compare_exchange_weak_explicit</code>. Dạng khả chuyển là vòng lặp thử lại{' '}
            <code>while (!cas(e) &amp;&amp; e == UNVISITED) {'{}'}</code>, không tốn gì khi không có lần thất bại giả nào.
          </p>
        }
      >
        <p>
          On CUDA, <code>cuda::atomic_ref::compare_exchange_strong</code> is strong. On HIP the same line compiles, but
          libhipcxx implements it at block and device scope with the weak built-in; the legacy <code>atomicCAS</code> is
          strong and returns the old value. The course's <code>.cu</code> files therefore use <code>atomicCAS</code>{' '}
          under <code>__HIPCC__</code>, as lab 12.1 uses legacy atomics. Metal has only{' '}
          <code>atomic_compare_exchange_weak_explicit</code>. The portable form is the retry loop{' '}
          <code>while (!cas(e) &amp;&amp; e == UNVISITED) {'{}'}</code>, which costs nothing when no failure is spurious.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md A9, B4, C4, C5)', '2026-11-20 (notes/today/ch18-facts.md, mục A9, B4, C4, C5)')}>
        <Translated
          vi={
            <p>
              Việc GPU AMD hay Apple có bao giờ làm một compare-and-swap yếu thất bại giả hay không thì không được tài liệu
              hóa: unverified. Cách viết đầy đủ của giá trị canh là <code>cuda::std::numeric_limits&lt;unsigned int&gt;::max()</code>{' '}
              từ <code>&lt;cuda/std/limits&gt;</code>; MSL có <code>numeric_limits&lt;uint&gt;::max()</code> và{' '}
              <code>UINT_MAX</code>.
            </p>
          }
        >
          <p>
            Whether AMD or Apple GPUs ever fail a weak compare-and-swap spuriously is not documented: unverified. The
            sentinel's full spelling is <code>cuda::std::numeric_limits&lt;unsigned int&gt;::max()</code> from{' '}
            <code>&lt;cuda/std/limits&gt;</code>; MSL has <code>numeric_limits&lt;uint&gt;::max()</code> and{' '}
            <code>UINT_MAX</code>.
          </p>
        </Translated>
      </Today>
      <VisitOnBackends />
      <table className="data wide" data-backends style={{ maxWidth: '58rem', tableLayout: 'fixed', width: '100%', overflowWrap: 'anywhere' }}>
        <colgroup>
          <col style={{ width: '19%' }} />
          <col style={{ width: '27%' }} />
          <col style={{ width: '27%' }} />
          <col style={{ width: '27%' }} />
        </colgroup>
        <thead>
          <tr><th><Translated vi="Tính năng">Feature</Translated></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Compare-and-swap thăm đúng một lần">Visit-once compare-and-swap</Translated></td>
            <td>
              <Translated vi={<><TC>compare_exchange_strong</TC> trên một <TC>atomic_ref</TC> (mạnh); <TC>atomicCAS</TC> kiểu cũ</>}>
                <TC>compare_exchange_strong</TC> on an <TC>atomic_ref</TC> (strong); legacy <TC>atomicCAS</TC>
              </Translated>
            </td>
            <td>
              <Translated vi={<>dạng mạnh của libhipcxx được hạ xuống built-in dạng yếu ở phạm vi khối và thiết bị; <TC>atomicCAS</TC> kiểu cũ là dạng mạnh</>}>
                libhipcxx's strong form lowered to the weak built-in at block and device scope; legacy <TC>atomicCAS</TC> strong
              </Translated>
            </td>
            <td>
              <Translated vi={<>chỉ có <TC>atomic_compare_exchange_weak_explicit</TC>; thất bại giả không được tài liệu hóa theo hướng nào</>}>
                <TC>atomic_compare_exchange_weak_explicit</TC> only; spurious failure not documented either way
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Giá trị expected sau một compare-and-swap thất bại">Expected value after a failed compare-and-swap</Translated></td>
            <td><Translated vi="bị ghi đè bằng giá trị hiện tại">overwritten with the current value</Translated></td>
            <td><Translated vi="như trên">same</Translated></td>
            <td><Translated vi="như trên">same</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Bộ đếm frontier toàn cục">Global frontier counter</Translated></td>
            <td>
              <Translated vi={<><TC>fetch_add(1, relaxed)</TC> trên một <TC>atomic_ref&lt;…, device&gt;</TC></>}>
                <TC>fetch_add(1, relaxed)</TC> on an <TC>atomic_ref&lt;…, device&gt;</TC>
              </Translated>
            </td>
            <td><Translated vi="như trên, thông qua libhipcxx">same, through libhipcxx</Translated></td>
            <td>
              <Translated vi={<><TC>atomic_fetch_add_explicit</TC> trên một <TC>device atomic_uint*</TC></>}>
                <TC>atomic_fetch_add_explicit</TC> on a <TC>device atomic_uint*</TC>
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Giá trị canh unsigned lớn nhất">Max-unsigned sentinel</Translated></td>
            <td>
              <Translated vi={<><TC>numeric_limits&lt;unsigned int&gt;::max()</TC> từ <TC>cuda::std</TC></>}>
                <TC>numeric_limits&lt;unsigned int&gt;::max()</TC> from <TC>cuda::std</TC>
              </Translated>
            </td>
            <td>
              <Translated vi={<>như trên, thông qua libhipcxx; hoặc <TC>UINT_MAX</TC></>}>
                same, through libhipcxx; or <TC>UINT_MAX</TC>
              </Translated>
            </td>
            <td>
              <Translated vi={<><TC>numeric_limits&lt;uint&gt;::max()</TC> hoặc <TC>UINT_MAX</TC></>}>
                <TC>numeric_limits&lt;uint&gt;::max()</TC> or <TC>UINT_MAX</TC>
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS183['ch18/18.3/q-weak-loop']} />
      <Translated
        vi={
          <p>
            Một bộ đếm mà mọi luồng đều chạm tới, và một lần khởi chạy cho mỗi mức: đó là hai chi phí còn lại.{' '}
            <a href={href('ch18', '18.6')}>Trang sau</a> loại bỏ chi phí thứ nhất bằng một hàng đợi riêng của mỗi khối và chi
            phí thứ hai bằng một rào chắn trên toàn lưới.
          </p>
        }
      >
        <p>
          One counter that every thread hits, and one launch per level: those are the two costs left. The{' '}
          <a href={href('ch18', '18.6')}>next page</a> removes the first with a queue private to each block and the second
          with a barrier across the grid.
        </p>
      </Translated>
    </>
  );
}

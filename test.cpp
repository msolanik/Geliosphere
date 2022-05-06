#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/signal.h>
#include <sys/types.h>

#define DESIRED_ADDRESS "127.0.0.1"
#define DESIRED_PORT 3500

int main()
{
    int pid = fork();
    switch (pid)
    {
        case 0: 
            execl("./Geliosphere", "-I");           
            break;
    
        default:
            pause();
            struct sockaddr_in addr = {0};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(DESIRED_PORT); /*converts short to
                                                        short with network byte order*/
            addr.sin_addr.s_addr = inet_addr(DESIRED_ADDRESS);

            int sock = socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (sock == -1) {
                perror("Socket creation error");
                return;
            }
            if (connect(sock, (struct sockaddr*) &addr, sizeof(addr)) == -1) {
                perror("Connection error");
                close(sock);
                return;
            }
            if (send(sock, "./Geliosphere -q", 15, 0) == -1) {
                perror("Send error");
                return EXIT_FAILURE;
            }
            break;
    }
    
    return 0;
}
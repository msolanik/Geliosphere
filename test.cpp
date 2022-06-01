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

void test(int signum)
{
    printf("Received signal %d\n", signum);
}

int main()
{
    signal(SIGUSR1, test);
    signal(SIGUSR2, test);
    int pid = fork();
    switch (pid)
    {
        case 0: 
            execl("./Geliosphere", "./Geliosphere", "-I", NULL);           
            break;
    
        default:
            pause();
            struct sockaddr_in addr = {0};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(DESIRED_PORT); /*converts short to
                                                        short with network byte order*/
            addr.sin_addr.s_addr = inet_addr(DESIRED_ADDRESS);

            int sock = socket (AF_INET, SOCK_STREAM, 0);
            if (sock == -1) {
                printf("Cannot create socket\n");
                perror("Socket creation error");
                return -1;
            }
            if (connect(sock, (struct sockaddr*) &addr, sizeof(addr)) == -1) {
                printf("Cannot connect\n");
                perror("Connection error");
                close(sock);
                return -1;
            }
            printf("Here1\n");
            if (send(sock, "./Geliosphere -B -N 1", 1024, 0) == -1) {
                printf("Cannot send\n");
                perror("Send error");
                return EXIT_FAILURE;
            }
            printf("Here2\n");
            pause();
            if (send(sock, "./Geliosphere -F -N 1", 1024, 0) == -1) {
                printf("Cannot send\n");
                perror("Send error");
                return EXIT_FAILURE;
            }
            printf("Here3\n");
            pause();
            if (send(sock, "./Geliosphere -Q", 1024, 0) == -1) {
                printf("Cannot send\n");
                perror("Send error");
                return EXIT_FAILURE;
            }
            printf("Here4\n");
            pause();
            printf("Here5\n");
            break;
    }
    
    printf("Ending testing program\n");
    return 0;
}
#include <stdio.h>
#define MAXLINE 1000

int my_getline(char line[], int maxline);
void copy(char to[], char from[]);

int main(){
    int len;
    int max;
    char line[MAXLINE];
    char longest_line[MAXLINE];

    max = 0;
    while((len = my_getline(line, MAXLINE)) > 0){
        if(len > max){
            max = len;
            copy(longest_line, line);
        }
    }
    if(max > 0){
        printf("%s", longest_line);
    }
    return 0;
}

int my_getline(char s[], int lim){
    int c, i;

    for(i=0; i<lim-1 && (c=getchar()) != EOF && c!='\n'; i++){
        s[i] = c;
    }

    if(c == '\n'){
        s[i] = c;
        i++;
    }
    s[i] = '\0';
    printf("this one was %d chars long!\n", i);
    return i;
}

void copy(char to[], char from[]){
    int i;

    while((to[i] = from[i]) != '\0'){
        i++;
    }

}
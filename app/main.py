import score

def main():
    #get user input
    user_in = 'a,b,c,d,e' #change this to take input from stdin
    user_in = [x.strip() for x in user_in.strip().split(",")]
    scores = score.get_score(user_in)


if __name__ == '__main__':
    main()

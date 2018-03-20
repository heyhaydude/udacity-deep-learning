import time
import util.parser as parser

def mean(s, games_played):
    return sum(s.keys()) / max(games_played, 1)


def save_to_file(game_name, max_score, games_played, frame_iterations, scores, training, start_time):
    """ TODO:  Move params to Stats object or dict"""
    session_minutes = (time.time() - start_time) / 60
    stats = "\n\nMax Score: {}\nGames Played: {}\nFrame Iterations: {}\n\nScores:\n{}\nTraining: {}\nSession Time: {:.2f} minutes\n\n" \
            .format(max_score, games_played, frame_iterations, parser.sorted_dict2str(scores), training, session_minutes)
    stats += "Mean: {}\n".format(mean(scores, games_played))
    stats += "="*40
    stats += "\n"
    f = open("statistics/stats_{}.txt".format(game_name), "a")
    f.write(stats)
    f.close()

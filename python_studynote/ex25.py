# -*- coding:utf-8 -*-.
def break_word(stuff):
	"""This Function will break up world for us"""
	words=stuff.split(' ')
	return words
print break_word("I will do this Job.")

def sort_words(words):
	"""Sorts the words"""
	return sorted(words)
print sort_words(break_word("I will do this Job."))
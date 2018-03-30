use 5.26.0;

use warnings;
use strict;

my $first_name = 'João Pedro';
my $last_name = "Andrade";

say $first_name . ' ' . $last_name;
say '$first_name $last_name';
say "$first_name $last_name";
say qq{"My name is: $first_name $last_name and I'm here to avenge my father"};


my $quote =<<'FLIGHT_QUOTE';

    For once you have tasted flight, 
    you will forever walk the earth 
    with your eyes turned skyward, 
    for there you have been, 
    and there you will always long to return.

FLIGHT_QUOTE

say $quote;

my @values = (4, 5, 6);
say $values[0];

my @list_of_strings = qw {"each of these words is an element of the list"};

say "The following should be words: $list_of_strings[3]";


sub lets_check_unless {
    return 'WHAT I AM SUPPOSED TO DO WITHOUT AN ARGUMENT' unless @_;

    for my $thing (@_) {say $thing}
};

say(lets_check_unless());
lets_check_unless($first_name);

# how about this crap

foreach (1 .. 5){
    say "$_ * $_ = ", $_ * $_;
};

# can also be

say "$_ + $_^2 = ", $_ + $_ ** 2 for 1 .. 5;  # this is fun and worrying

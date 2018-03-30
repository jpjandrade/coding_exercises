use 5.26.0;

use warnings;
use strict;
use Data::Dumper;


my $zipcode = 1071;

# it's just scalar, coerced at will from int to string
my $address = $zipcode . " VS, Amsterdam";

# it's also evaluated as true, of course lol
say $address if $zipcode;

# But perl is perl
my $false_variable = '0';
my $true_variable = '0.000';

say "This is printed because '0' is false" unless $false_variable;
say "BUT! This is printed because '0.0' is true ¯\\_(ツ)_/¯" if $true_variable;

# also there's one thing
my $lol_string = "LOK"; 

say $lol_string++; # hmmm... doesn't work. passes value then adds
say ++$lol_string; # nailed it

# array and conversion

my @cats = ("Garfield", "Tom", "Felix");
my $cat_count = @cats; # COERCION
say "I have $cat_count cats";

# you can coerce things to arrays
my @list_of_zipcodes = $zipcode;
@list_of_zipcodes[2] = 1075;

say Dumper(@list_of_zipcodes); # check the implicit undef




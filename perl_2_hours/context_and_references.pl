# structure

# print is in list context

my @arr = ("Alpha", "Beta", "Goo");
my $scalar = "-X-";
print @arr;              # "AlphaBetaGoo";
print $scalar, @arr, 98; # "-X-AlphaBetaGoo98";


print "\n\n";

# reversing shit

# list are reversed in items
# scalars are reversed as strings
print "reversing stuff\n\n";
print reverse "hello world"; # "hello world"
print "\n\n";
my $string = reverse "hello world";
print $string; # "dlrow olleh"

print "\n\n";

# array references

my @colours = ("Red", "Orange", "Yellow", "Green", "Blue");
my $arrayRef = \@colours;

print $colours[0];       # direct array access
print ${ $arrayRef }[0]; # use the reference to get to the array
print $arrayRef->[0];

print "\n\n";

my %atomicWeights = ("Hydrogen" => 1.008, "Helium" => 4.003, "Manganese" => 54.94);
my $hashRef = \%atomicWeights;

print $atomicWeights{"Helium"}; # direct hash access
print ${ $hashRef }{"Helium"};  # use a reference to get to the hash
print $hashRef->{"Helium"};     # exactly the same thing - this is very common

print "\n\n";
use strict;
use warnings;

my $undef = undef;
print $undef;

my $undef2;
print $undef2;

my $num = 400.5;
print $num;
print "\n";

# concatenation

my $str = 'World';
print "Hello ".$str."\n";

my @arr = (
    "omg",
    "I",
    "just",
    "created",
    "an",
    "array"
    );

print $arr[0]."\n";

print "This array has ".(scalar @arr)." elements\n";

my %scientists = (
    "Newton" => "Isaac",
    "Einstein" => "Albert",
    "Darwin" => "Charles"
    );

print "Getting scientists last names:\n";
print $scientists{"Newton"}."\n";
print $scientists{"Einstein"}."\n";
print $scientists{"Darwin"}."\n";
print $scientists{"Dyson"}."\n";

print "@arr"."\n";
print "%scientists"."\n";

my $data = "orange";
my @data = ("purple");
my %data = ( "0" => "blue");

print $data."\n";      # "orange"
print $data[0]."\n";   # "purple"
print $data["0"]."\n"; # "purple"
print $data{0}."\n";   # "blue"
print $data{"0"}."\n"; # "blue"

my %what = ("Alpha", "Beta", "Gamma", "Pie");
print $what{"Alpha"}."\n";
print $what{"Beta"}."\n";
print $what{"Gamma"}."\n";
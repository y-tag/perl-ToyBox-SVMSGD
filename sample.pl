#/usr/bin/perl

use strict;
use warnings;

use Data::Dumper;

use lib('lib');
use ToyBox::SVMSGD;

my $svm = ToyBox::SVMSGD->new();

my @pos_data = qw(a b ab bb aaab);
my @neg_data = qw(cc d cd ddd ccd);

foreach my $d (@pos_data) {
    my $tmp = make_attributes($d);
    $svm->add_instance(attributes => $tmp, label => 'positive');
}
foreach my $d (@neg_data) {
    my $tmp = make_attributes($d);
    $svm->add_instance(attributes => $tmp, label =>'negative');
}

$svm->train(progress_cb => 'verbose');

foreach my $d qw(abcdd abdf) {
    my $tmp   = make_attributes($d);
    my $score = $svm->predict(attributes => $tmp);
    print Dumper($d, $score);
}

print Dumper($svm->labels);


sub make_attributes {
    my $data = shift;

    my $attributes = {};
    foreach my $chr (split(//, $data)) {
        $attributes->{$chr}++;
    }

    $attributes;
}

